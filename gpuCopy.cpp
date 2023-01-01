#include "gpuCopy.h"

#include <array>
#include <chrono>
#include <execution>
#include <expected>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <ranges>
#include <source_location>
#include <span>

constexpr void BAIL_ON_BAD_RESULT(auto result,
                                  std::source_location location = std::source_location::current())
{
    if (result != VK_SUCCESS)
    {
        std::cout << "Failure at line " << location.line() << " in " << location.file_name()
                  << "\n";
        exit(-1);
    }
}

namespace
{
std::expected<uint32_t, VkResult> getBestComputeQueue(
    const vk::raii::PhysicalDevice& physicalDevice)
{
    const auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
    // first try and find a queue that has just the compute bit set
    auto computeWithoutGraphics = [](const auto& properties) {
        // mask out the sparse binding bit that we aren't caring about (yet!) and
        // the transfer bit
        const auto maskedFlags =
            (~(vk::QueueFlagBits::eTransfer | vk::QueueFlagBits::eSparseBinding) &
             properties.queueFlags);
        return !(vk::QueueFlagBits::eGraphics & maskedFlags) &&
               (vk::QueueFlagBits::eCompute & maskedFlags);
    };

    const auto optimalQueue = std::ranges::find_if(queueFamilyProperties, computeWithoutGraphics);
    if (optimalQueue != queueFamilyProperties.end())
    {
        return std::distance(queueFamilyProperties.begin(), optimalQueue);
    }

    // lastly get any queue that'll work for us
    auto hasCompute = [](const auto& properties) -> bool {
        const auto maskedFlags =
            (~(vk::QueueFlagBits::eTransfer | vk::QueueFlagBits::eSparseBinding) &
             properties.queueFlags);
        return (vk::QueueFlagBits::eCompute & maskedFlags) && true;
    };

    const auto computeQueue = std::ranges::find_if(queueFamilyProperties, hasCompute);
    if (computeQueue != queueFamilyProperties.end())
    {
        return std::distance(queueFamilyProperties.begin(), computeQueue);
    }

    return std::unexpected{VK_ERROR_INITIALIZATION_FAILED};
}

auto div_up(uint32_t x, uint32_t y)
{
    return (x + y - 1u) / y;
}

size_t nextPowerOf2(size_t n)
{
    size_t v = 1;
    while (v < n)
    {
        v *= 2;
    }
    return v;
}

auto elapsedSince(const auto start)
{
    return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() -
                                                     start)
        .count();
}

auto getSpirvFromFile(const std::string_view filePath)
{
    using spirv_t = uint32_t;
    using file_t = char;
    std::vector<file_t> spvBuffer;
    {
        std::ifstream spirvFile(filePath.data(), std::ios::binary | std::ios::ate);
        const auto size = spirvFile.tellg();
        spirvFile.seekg(0, std::ios::beg);

        spvBuffer.resize(size);

        if (!spirvFile.read(spvBuffer.data(), size))
        {
            std::cout << "Could not read spv file"
                      << "\n";
            exit(1);
        }
    }
    constexpr auto sizeDivisor = sizeof(spirv_t) / sizeof(file_t);
    static_assert(sizeDivisor != 0);

    spvBuffer.resize(sizeDivisor * div_up(spvBuffer.size(), sizeDivisor));

    std::vector<spirv_t> spirvFromFile(spvBuffer.size() / sizeDivisor);
    assert(spvBuffer.size() == spirvFromFile.size() * sizeDivisor);

    const auto start = reinterpret_cast<spirv_t*>(spvBuffer.data());
    std::copy(start, start + spirvFromFile.size(), spirvFromFile.data());
    return spirvFromFile;
}

const static auto spirv = getSpirvFromFile("copy.comp.spv");
using bufferData_t = int32_t;

uint32_t requiredMemorySize(const uint32_t singleBufferLength)
{
    const uint32_t bufferSize = sizeof(bufferData_t) * singleBufferLength;
    const auto memorySize = bufferSize * 2;
    return memorySize;
}

auto mapAllRequiredMemory(const auto& memory, const uint32_t singleBufferLength)
{
    auto* payload =
        static_cast<bufferData_t*>(memory.mapMemory(0, requiredMemorySize(singleBufferLength)));
    if (!payload)
    {
        BAIL_ON_BAD_RESULT(VK_ERROR_OUT_OF_HOST_MEMORY);
    }
    return std::span(payload, singleBufferLength * 2);
}

void generateRandomDataOnDevice(const vk::raii::DeviceMemory& memory,
                                const uint32_t singleBufferLength)
{
    auto payloadSpan = mapAllRequiredMemory(memory, singleBufferLength);
    auto inputSpan = payloadSpan.subspan(0, singleBufferLength);
    auto rng = std::mt19937(std::chrono::steady_clock::now().time_since_epoch().count());
    std::generate(inputSpan.begin(), inputSpan.end(), rng);
    const auto outputSpan = payloadSpan.subspan(inputSpan.size(), inputSpan.size());

    if (std::ranges::equal(inputSpan, outputSpan))
    {
        std::cout << "The memory already had equal values"
                  << "\n";
    }

    memory.unmapMemory();
}

uint32_t getLocalGroupSize(const vk::raii::PhysicalDevice& physDev, const uint32_t bufferLength)
{
    const auto props2 =
        physDev
            .getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceSubgroupProperties>();
    const auto subGroupProps = props2.get<vk::PhysicalDeviceSubgroupProperties>();

    std::cout << "Subgroup Size: " << subGroupProps.subgroupSize << "\n";

    // Adjust the subgroup size to avoid invoking too many workgroups
    const auto maxWorkGroupCountX = physDev.getProperties().limits.maxComputeWorkGroupCount[0];

    const auto subgroupMultiplier =
        subGroupProps.subgroupSize * maxWorkGroupCountX > bufferLength
            ? 1
            : nextPowerOf2(bufferLength / (maxWorkGroupCountX * subGroupProps.subgroupSize) + 1);

    const uint32_t localGroupSize = subGroupProps.subgroupSize * subgroupMultiplier;

    std::cout << "Local Group Size used: " << localGroupSize << "\n";
    return localGroupSize;
}

auto getDevice(const vk::raii::PhysicalDevice& physDev, const auto queueFamilyIndex)
{
    constexpr std::array queuePrioritory = {1.0f};
    const auto deviceQueueCreateInfo =
        vk::DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), queueFamilyIndex, queuePrioritory);

    const std::array queueInfos = {deviceQueueCreateInfo};
    const auto deviceCreateInfo = vk::DeviceCreateInfo(vk::DeviceCreateFlags(), queueInfos);

    return vk::raii::Device(physDev, deviceCreateInfo);
}

auto getDeviceMemory(const vk::raii::Device& device,
                     const vk::PhysicalDeviceMemoryProperties& props, const uint32_t memorySize)
{
    const auto memoryTypeIndex = [&props, memorySize]() -> std::optional<size_t> {
        for (const auto& k : std::views::iota(0u, props.memoryTypeCount))
        {
            std::cout << to_string(props.memoryTypes[k].propertyFlags) << "\n";
            if ((vk::MemoryPropertyFlagBits::eHostVisible & props.memoryTypes[k].propertyFlags) &&
                (vk::MemoryPropertyFlagBits::eHostCoherent & props.memoryTypes[k].propertyFlags) &&
                (vk::MemoryPropertyFlagBits::eDeviceLocal & props.memoryTypes[k].propertyFlags) &&
                (memorySize < props.memoryHeaps[props.memoryTypes[k].heapIndex].size))
            {
                return k;
            }
        }
        return {};
    }();

    if (!memoryTypeIndex)
    {
        BAIL_ON_BAD_RESULT(VK_ERROR_OUT_OF_HOST_MEMORY);
    }

    std::cout << "Memory type index: " << *memoryTypeIndex << "\n";

    const vk::MemoryAllocateInfo memoryAllocateInfo(memorySize, *memoryTypeIndex);

    return vk::raii::DeviceMemory(device, memoryAllocateInfo);
}

auto makeDescriptorSetLayout(const auto& device)
{
    constexpr std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1,
                                       vk::ShaderStageFlagBits::eCompute, nullptr),
        vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1,
                                       vk::ShaderStageFlagBits::eCompute, nullptr)};

    auto descriptorSetLayout = vk::raii::DescriptorSetLayout(
        device, vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), bindings));
    return descriptorSetLayout;
}

auto makePipelineLayout(const auto& device, const auto& descriptorSetLayout)
{
    const auto pipelineCreateInfo =
        vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), *descriptorSetLayout);
    return vk::raii::PipelineLayout(device, pipelineCreateInfo);
}

auto makePipeline(const auto& device, const auto& pipelineLayout, const uint32_t localGroupSize)
{
    const auto shaderModule = vk::raii::ShaderModule(
        device, vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(), spirv));
    const auto specializationEntry =
        vk::SpecializationMapEntry({.constantID = 0, .offset = 0, .size = sizeof(localGroupSize)});
    const auto specializationInfo =
        vk::SpecializationInfo(1, &specializationEntry, sizeof(localGroupSize), &localGroupSize);
    const auto shaderStageCreateInfo = vk::PipelineShaderStageCreateInfo(
        vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, *shaderModule,
        "main", &specializationInfo);

    const auto computePipelineCreateInfo = vk::ComputePipelineCreateInfo(
        vk::PipelineCreateFlags(), shaderStageCreateInfo, *pipelineLayout);

    auto pipeline = vk::raii::Pipeline(device, nullptr, computePipelineCreateInfo);
    return pipeline;
}

auto makeDescriptorPool(const auto& device)
{
    constexpr auto DescriptorCount = 2;
    constexpr auto descriptorPoolSize =
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, DescriptorCount);
    constexpr std::array descriptorPoolSizeArray = {descriptorPoolSize};
    const auto descriptorPoolCreateInfo = vk::DescriptorPoolCreateInfo(
        vk::DescriptorPoolCreateFlags() | vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1,
        descriptorPoolSizeArray);

    assert(descriptorPoolCreateInfo.flags & vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);

    return vk::raii::DescriptorPool(device, descriptorPoolCreateInfo);
}

auto allocateDescriptorSet(const auto& device, const auto& descriptorPool,
                           const auto& descriptorSetLayout)
{
    const vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo(*descriptorPool,
                                                                  *descriptorSetLayout);

    auto descriptorSets = device.allocateDescriptorSets(descriptorSetAllocateInfo);
    assert(descriptorSets.size() == 1);
    vk::raii::DescriptorSet single = std::move(descriptorSets[0]);
    return single;
}

auto makeBoundBuffers(const auto& device, const auto& memory, const uint32_t queueFamilyIndex,
                      const uint32_t bufferLength)
{
    const std::array indices = {queueFamilyIndex};
    const auto bufferSize = requiredMemorySize(bufferLength) / 2;
    const auto bufferCreateInfo = vk::BufferCreateInfo(vk::BufferCreateFlags(), bufferSize,
                                                       vk::BufferUsageFlagBits::eStorageBuffer,
                                                       vk::SharingMode::eExclusive, indices);
    auto in_buffer = vk::raii::Buffer(device, bufferCreateInfo);

    auto out_buffer = vk::raii::Buffer(device, bufferCreateInfo);

    in_buffer.bindMemory(*memory, 0);

    out_buffer.bindMemory(*memory, bufferSize);
    return std::make_pair(std::move(in_buffer), std::move(out_buffer));
}

void updateDescriptorSetsWithBufferInfo(const auto& device, const auto& in_buffer,
                                        const auto& out_buffer, const auto& descriptorSet)
{
    const auto in_descriptorBufferInfo = vk::DescriptorBufferInfo(*in_buffer, 0, VK_WHOLE_SIZE);
    const auto out_descriptorBufferInfo = vk::DescriptorBufferInfo(*out_buffer, 0, VK_WHOLE_SIZE);

    constexpr auto inBindingIndex = 0;
    constexpr auto outBindingIndex = 1;
    const std::array writeDescriptorSet = {
        vk::WriteDescriptorSet(*descriptorSet, inBindingIndex, 0, 1,
                               vk::DescriptorType::eStorageBuffer, nullptr,
                               &in_descriptorBufferInfo),
        vk::WriteDescriptorSet(*descriptorSet, outBindingIndex, 0, 1,
                               vk::DescriptorType::eStorageBuffer, nullptr,
                               &out_descriptorBufferInfo)};
    device.updateDescriptorSets(writeDescriptorSet, {});
}

auto makeAndRecordCommandBuffer(const auto& device, const auto& pipeline,
                                const auto& pipelineLayout, const auto& descriptorSet,
                                const uint32_t queueFamilyIndex, const size_t groupCountX)
{
    auto commandPool = vk::raii::CommandPool(
        device, vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlags(), queueFamilyIndex));
    constexpr auto commandBuffersCount = 1;
    auto commandBuffers = vk::raii::CommandBuffers(
        device, vk::CommandBufferAllocateInfo(*commandPool, vk::CommandBufferLevel::ePrimary,
                                              commandBuffersCount));

    auto& commandBuffer = commandBuffers.front();
    commandBuffer.begin(
        vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eRenderPassContinue));
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0,
                                     *descriptorSet, nullptr);

    commandBuffer.dispatch(groupCountX, 1, 1);
    commandBuffer.end();
    // apparently the order here is important: there must be a valid command pool by the time the
    // command buffer is destroyed (so the command buffer must be destroyed first)
    return std::make_pair(std::move(commandPool), std::move(commandBuffer));
}
} // namespace

int copyUsingDevice(const vk::raii::PhysicalDevice& physDev, const uint32_t bufferLength)
{
    const auto localGroupSize = getLocalGroupSize(physDev, bufferLength);
    const auto queueFamilyIndex = getBestComputeQueue(physDev);
    if (!queueFamilyIndex)
    {
        BAIL_ON_BAD_RESULT(queueFamilyIndex.error());
    }

    const auto device = getDevice(physDev, *queueFamilyIndex);

    const auto memorySize = requiredMemorySize(bufferLength);

    const auto memory = getDeviceMemory(device, physDev.getMemoryProperties(), memorySize);

    const auto clock = std::chrono::high_resolution_clock();
    {
        const auto start = clock.now();
        generateRandomDataOnDevice(memory, bufferLength);
        const auto elapsed = elapsedSince(start);
        std::cout << "Random data generation duration: " << elapsed << "\n";
    }

    std::cout << to_string(memory.debugReportObjectType) << "\n";

    const auto descriptorSetLayout = makeDescriptorSetLayout(device);
    const auto pipelineLayout = makePipelineLayout(device, descriptorSetLayout);

    const auto pipeline = makePipeline(device, pipelineLayout, localGroupSize);

    const auto descriptorPool = makeDescriptorPool(device);

    const auto descriptorSet = allocateDescriptorSet(device, descriptorPool, descriptorSetLayout);
    // Create in/out buffers with descriptors and bind to memory
    const auto [in_buffer, out_buffer] =
        makeBoundBuffers(device, memory, *queueFamilyIndex, bufferLength);

    updateDescriptorSetsWithBufferInfo(device, in_buffer, out_buffer, descriptorSet);

    const auto [commandPool, commandBuffer] =
        makeAndRecordCommandBuffer(device, pipeline, pipelineLayout, descriptorSet,
                                   *queueFamilyIndex, bufferLength / localGroupSize);
    constexpr auto queueIndex = 0;
    const auto queue = vk::raii::Queue(device, *queueFamilyIndex, queueIndex);

    constexpr size_t numberOfQueueSubmissions = 10;
    {
        const auto start = clock.now();
        std::ranges::for_each(std::views::iota(0u, numberOfQueueSubmissions), [&](auto) {
            queue.submit(vk::SubmitInfo(nullptr, nullptr, *commandBuffer));
            queue.waitIdle();
        });
        const auto elapsed = elapsedSince(start);
        std::cout << "Duration of copying data " << numberOfQueueSubmissions
                  << " times on GPU: " << elapsed << "\n";
    }

    const auto outputSpan = mapAllRequiredMemory(memory, bufferLength);

    assert(memorySize / sizeof(outputSpan.front()) == outputSpan.size());
    const auto frontHalf = outputSpan.subspan(0, outputSpan.size() / 2);
    const auto backHalf = outputSpan.subspan(frontHalf.size(), frontHalf.size());

    // Let's just assume that if something went wrong, the first few front and back values wouldn't
    // match. This saves us from having to check the whole range every time.
    const size_t numElementsToCheck = std::min(100u, bufferLength);
    const auto firstElementsEqual =
        std::ranges::equal(frontHalf.first(numElementsToCheck), backHalf.first(numElementsToCheck));
    const auto lastElementsEqual =
        std::ranges::equal(frontHalf.last(numElementsToCheck), backHalf.last(numElementsToCheck));
    if (!(firstElementsEqual && lastElementsEqual))
    {
        const auto [p1, p2] = std::ranges::mismatch(frontHalf, backHalf);

        if (p1 != frontHalf.end())
        {
            std::cout << "Bad at " << std::distance(frontHalf.begin(), p1) << "\n";
        }
        if (p2 != backHalf.end())
        {
            std::cout << "Bad at " << std::distance(backHalf.begin(), p2) << "\n";
        }
    }

    return 0;
}
