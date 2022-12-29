#include "gpuCopy.h"

#include <array>
#include <expected>
#include <fstream>
#include <iostream>
#include <optional>
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
std::expected<uint32_t, VkResult> getBestComputeQueue(const auto& physicalDevice)
{
    const auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
    using queueIndex_t = uint32_t;
    // first try and find a queue that has just the compute bit set
    const auto queueIndices =
        std::views::iota(queueIndex_t(0), queueIndex_t(queueFamilyProperties.size()));
    for (const auto queueIndex : queueIndices)
    {
        const auto& prop = queueFamilyProperties[queueIndex];
        // mask out the sparse binding bit that we aren't caring about (yet!) and
        // the transfer bit
        const auto maskedFlags =
            (~(vk::QueueFlagBits::eTransfer | vk::QueueFlagBits::eSparseBinding) & prop.queueFlags);

        if (!(vk::QueueFlagBits::eGraphics & maskedFlags) &&
            (vk::QueueFlagBits::eCompute & maskedFlags))
        {
            return queueIndex;
        }
    }

    // lastly get any queue that'll work for us
    for (const auto queueIndex : queueIndices)
    {
        const auto& prop = queueFamilyProperties[queueIndex];
        // mask out the sparse binding bit that we aren't caring about (yet!) and
        // the transfer bit
        const auto maskedFlags =
            (~(vk::QueueFlagBits::eTransfer | vk::QueueFlagBits::eSparseBinding) & prop.queueFlags);

        if (vk::QueueFlagBits::eCompute & maskedFlags)
        {
            return queueIndex;
        }
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

auto getSpirvFromFile(const std::string_view filePath)
{
    using spirv_t = uint32_t;
    using file_t = char;
    std::vector<file_t> spvBuffer;
    {
        std::ifstream spirvFile(filePath.data(), std::ios::binary | std::ios::ate);
        std::streamsize size = spirvFile.tellg();
        spirvFile.seekg(0, std::ios::beg);

        spvBuffer.resize(size);

        if (!spirvFile.read(spvBuffer.data(), size))
        {
            std::cout << "Could not read spv file"
                      << "\n";
            exit(1);
        }
    }
    std::vector<spirv_t> spirvFromFile;
    constexpr auto sizeDivisor = sizeof(spirv_t) / sizeof(file_t);
    static_assert(sizeDivisor != 0);

    spvBuffer.resize(sizeDivisor * div_up(spvBuffer.size(), sizeDivisor));
    spirvFromFile.resize(spvBuffer.size() / sizeDivisor);
    const auto start = reinterpret_cast<spirv_t*>(spvBuffer.data());
    std::copy(start, start + spirvFromFile.size(), spirvFromFile.data());
    return spirvFromFile;
}

const static auto spirv = getSpirvFromFile("copy.comp.spv");

} // namespace

int copyUsingDevice(const vk::raii::PhysicalDevice& physDev, const uint32_t bufferLength)
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

    const auto queueFamilyIndex = getBestComputeQueue(physDev);
    if (!queueFamilyIndex)
    {
        BAIL_ON_BAD_RESULT(queueFamilyIndex.error());
    }

    constexpr std::array queuePrioritory = {1.0f};
    const auto deviceQueueCreateInfo =
        vk::DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), *queueFamilyIndex, queuePrioritory);

    const std::array queueInfos = {deviceQueueCreateInfo};
    const auto deviceCreateInfo = vk::DeviceCreateInfo(vk::DeviceCreateFlags(), queueInfos);

    const auto device = vk::raii::Device(physDev, deviceCreateInfo);
    const auto props = physDev.getMemoryProperties();

    const uint32_t bufferSize = sizeof(bufferData_t) * bufferLength;
    const auto memorySize = bufferSize * 2;

    const auto memoryTypeIndex = [&props, memorySize]() -> std::optional<size_t> {
        for (const auto& k : std::views::iota(0u, props.memoryTypeCount))
        {
            std::cout << to_string(props.memoryTypes[k].propertyFlags) << "\n";
            if ((vk::MemoryPropertyFlagBits::eHostVisible & props.memoryTypes[k].propertyFlags) &&
                (vk::MemoryPropertyFlagBits::eHostCoherent & props.memoryTypes[k].propertyFlags) &&
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

    const vk::raii::DeviceMemory memory(device, memoryAllocateInfo);

    std::vector<bufferData_t> copyOfInputData;
    {
        auto* payload = static_cast<bufferData_t*>(memory.mapMemory(0, memorySize));
        if (!payload)
        {
            BAIL_ON_BAD_RESULT(VK_ERROR_OUT_OF_HOST_MEMORY);
        }
        auto payloadSpan = std::span(payload, memorySize / sizeof(*payload));
        std::ranges::for_each(payloadSpan, [](auto& elem) { elem = std::rand(); });
        /* std::ranges::for_each_n(payload, 10,
                                [](const auto& elem) { std::cout << elem << "\n"; }); */
        const auto inputSpan = payloadSpan.subspan(0, payloadSpan.size() / 2);
        const auto outputSpan = payloadSpan.subspan(inputSpan.size(), inputSpan.size());
        if (std::ranges::equal(inputSpan, outputSpan))
        {
            std::cout << "The memory already had equal values"
                      << "\n";
        }
        copyOfInputData.resize(payloadSpan.size());
        std::ranges::copy(payloadSpan, copyOfInputData.begin());
    }

    std::cout << to_string(memory.debugReportObjectType) << "\n";

    memory.unmapMemory();

    constexpr std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1,
                                       vk::ShaderStageFlagBits::eCompute, nullptr),
        vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1,
                                       vk::ShaderStageFlagBits::eCompute, nullptr)};

    const auto descriptorSetLayout = vk::raii::DescriptorSetLayout(
        device, vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), bindings));

    const auto pipelineLayout = vk::raii::PipelineLayout(
        device,
        vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), *descriptorSetLayout));

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

    const auto pipeline = vk::raii::Pipeline(device, nullptr, computePipelineCreateInfo);

    constexpr auto DescriptorCount = 2;
    constexpr auto descriptorPoolSize =
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, DescriptorCount);
    constexpr std::array descriptorPoolSizeArray = {descriptorPoolSize};
    const auto descriptorPoolCreateInfo = vk::DescriptorPoolCreateInfo(
        vk::DescriptorPoolCreateFlags() | vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1,
        descriptorPoolSizeArray);

    assert(descriptorPoolCreateInfo.flags & vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);

    const auto descriptorPool = vk::raii::DescriptorPool(device, descriptorPoolCreateInfo);

    const vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo(*descriptorPool,
                                                                  *descriptorSetLayout);

    const auto descriptorSets = device.allocateDescriptorSets(descriptorSetAllocateInfo);
    assert(descriptorSets.size() == 1);
    const auto& descriptorSet = descriptorSets[0];

    // Create in/out buffers with descriptors and bind to memory
    const std::array indices = {*queueFamilyIndex};
    const auto bufferCreateInfo = vk::BufferCreateInfo(vk::BufferCreateFlags(), bufferSize,
                                                       vk::BufferUsageFlagBits::eStorageBuffer,
                                                       vk::SharingMode::eExclusive, indices);
    const auto in_buffer = vk::raii::Buffer(device, bufferCreateInfo);

    const auto out_buffer = vk::raii::Buffer(device, bufferCreateInfo);

    in_buffer.bindMemory(*memory, 0);

    out_buffer.bindMemory(*memory, bufferSize);

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

    const auto commandPool = vk::raii::CommandPool(
        device, vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlags(), *queueFamilyIndex));

    constexpr auto commandBuffersCount = 1;
    const auto commandBuffers = vk::raii::CommandBuffers(
        device, vk::CommandBufferAllocateInfo(*commandPool, vk::CommandBufferLevel::ePrimary,
                                              commandBuffersCount));

    const auto& commandBuffer = commandBuffers.front();
    commandBuffer.begin(
        vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eRenderPassContinue));
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0,
                                     *descriptorSet, nullptr);

    commandBuffer.dispatch(bufferLength / localGroupSize, 1, 1);
    commandBuffer.end();

    constexpr auto queueIndex = 0;
    const auto queue = vk::raii::Queue(device, *queueFamilyIndex, queueIndex);

    std::ranges::for_each(std::views::iota(1, 2), [&](auto) {
        queue.submit(vk::SubmitInfo(nullptr, nullptr, *commandBuffer));
        queue.waitIdle();
    });

    const auto* payload =
        static_cast<bufferData_t*>(memory.mapMemory(0, memorySize, vk::MemoryMapFlags{0}));
    const auto outputSpan = std::span(payload, copyOfInputData.size());

    assert(memorySize / sizeof(*payload) == outputSpan.size());
    const auto frontHalf = outputSpan.subspan(0, outputSpan.size() / 2);
    const auto backHalf = outputSpan.subspan(frontHalf.size(), frontHalf.size());

    const auto [p1, p2] = std::ranges::mismatch(frontHalf, backHalf);
    if (p1 != frontHalf.end())
    {
        std::cout << "Bad at " << std::distance(frontHalf.begin(), p1) << "\n";
    }
    if (p2 != backHalf.end())
    {
        std::cout << "Bad at " << std::distance(backHalf.begin(), p2) << "\n";
    }

    const auto [i1, i2] = std::ranges::mismatch(copyOfInputData, outputSpan);
    if (i1 != copyOfInputData.end())
    {
        std::cout << "Input and Output differ at " << std::distance(copyOfInputData.begin(), i1)
                  << "/" << copyOfInputData.size() << "\n";
    }

    return 0;
}
