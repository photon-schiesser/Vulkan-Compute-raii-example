#include <array>
#include <fstream>
#include <iostream>
#include <optional>
#include <ranges>
#include <span>
#include <vector>

#define VULKAN_HPP_NO_SMART_HANDLE
#include "vulkan/vulkan.hpp"
#include "vulkan/vulkan_raii.hpp"

#define BAIL_ON_BAD_RESULT(result)                                                                 \
    if ((result) != VK_SUCCESS)                                                                    \
    {                                                                                              \
        fprintf(stderr, "Failure at %u %s\n", __LINE__, __FILE__);                                 \
        exit(-1);                                                                                  \
    }

namespace
{
std::pair<VkResult, std::optional<uint32_t>> getBestComputeQueue(const auto& physicalDevice)
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
            return {VK_SUCCESS, queueIndex};
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
            return {VK_SUCCESS, queueIndex};
        }
    }

    return {VK_ERROR_INITIALIZATION_FAILED, {}};
}
} // namespace

inline auto div_up(uint32_t x, uint32_t y)
{
    return (x + y - 1u) / y;
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
            spirvFile.close();
            exit(1);
        }
        spirvFile.close();
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

int main()
{
    constexpr vk::ApplicationInfo applicationInfo = []() {
        vk::ApplicationInfo temp;
        temp.pApplicationName = "Compute-Pipeline";
        temp.applicationVersion = 1;
        temp.pEngineName = nullptr;
        temp.engineVersion = 0;
        temp.apiVersion = VK_MAKE_VERSION(1, 0, 9);
        return temp;
    }();

    constexpr std::array layers = {"VK_LAYER_KHRONOS_validation"};
    const vk::InstanceCreateInfo instanceCreateInfo(vk::InstanceCreateFlags(), &applicationInfo,
                                                    layers.size(), layers.data());

    const vk::raii::Context context;
    const auto instance = vk::raii::Instance(context, instanceCreateInfo);

    constexpr uint32_t bufferLength = 16384 * 2 * 32;
    constexpr uint32_t bufferSize = sizeof(int32_t) * bufferLength;
    constexpr auto memorySize = bufferSize * 2;

    const auto spirvFromFile = getSpirvFromFile("copy.comp.spv");

    const auto physicalDevices = instance.enumeratePhysicalDevices();
    for (const auto& physDev : physicalDevices)
    {
        const auto [result, queueFamilyIndex] = getBestComputeQueue(physDev);
        if (!queueFamilyIndex)
        {
            BAIL_ON_BAD_RESULT(result);
            exit(1);
        }

        constexpr std::array queuePrioritory = {1.0f};
        const auto deviceQueueCreateInfo = vk::DeviceQueueCreateInfo(
            vk::DeviceQueueCreateFlags(), *queueFamilyIndex, queuePrioritory);

        const std::array queueInfos = {deviceQueueCreateInfo};
        const auto deviceCreateInfo = vk::DeviceCreateInfo(vk::DeviceCreateFlags(), queueInfos);

        const auto device = vk::raii::Device(physDev, deviceCreateInfo);
        const auto props = physDev.getMemoryProperties();

        const auto memoryTypeIndex = [&props]() -> std::optional<size_t> {
            for (const auto& k : std::views::iota(0u, props.memoryTypeCount))
            {
                std::cout << to_string(props.memoryTypes[k].propertyFlags) << "\n";
                if ((vk::MemoryPropertyFlagBits::eHostVisible &
                     props.memoryTypes[k].propertyFlags) &&
                    (vk::MemoryPropertyFlagBits::eHostCoherent &
                     props.memoryTypes[k].propertyFlags) &&
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

        std::vector<int32_t> copyOfInputData;
        {
            auto* payload = static_cast<int32_t*>(memory.mapMemory(0, memorySize));
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
            device,
            vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), bindings));

        const auto pipelineLayout = vk::raii::PipelineLayout(
            device,
            vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), *descriptorSetLayout));

        const auto shaderModule = vk::raii::ShaderModule(
            device, vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(), spirvFromFile));

        const auto shaderStageCreateInfo = vk::PipelineShaderStageCreateInfo(
            vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, *shaderModule,
            "main");

        const auto computePipelineCreateInfo = vk::ComputePipelineCreateInfo(
            vk::PipelineCreateFlags(), shaderStageCreateInfo, *pipelineLayout);

        const auto pipeline = vk::raii::Pipeline(device, nullptr, computePipelineCreateInfo);

        constexpr auto DescriptorCount = 2;
        constexpr auto descriptorPoolSize =
            vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, DescriptorCount);
        constexpr std::array descriptorPoolSizeArray = {descriptorPoolSize};
        const auto descriptorPoolCreateInfo = vk::DescriptorPoolCreateInfo(
            vk::DescriptorPoolCreateFlags() | vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            1, descriptorPoolSizeArray);

        assert(descriptorPoolCreateInfo.flags &
               vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);

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
        const auto out_descriptorBufferInfo =
            vk::DescriptorBufferInfo(*out_buffer, 0, VK_WHOLE_SIZE);

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
            vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0,
                                         *descriptorSet, nullptr);

        commandBuffer.dispatch(bufferLength/32, 1, 1);
        commandBuffer.end();

        constexpr auto queueIndex = 0;
        const auto queue = vk::raii::Queue(device, *queueFamilyIndex, queueIndex);

        queue.submit(vk::SubmitInfo(nullptr, nullptr, *commandBuffer));
        queue.waitIdle();

        const auto* payload =
            static_cast<int32_t*>(memory.mapMemory(0, memorySize, vk::MemoryMapFlags{0}));
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
    }
    return 0;
}