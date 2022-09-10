#include "makeSpirvCode.hpp"

#include <array>
#include <iostream>
#include <optional>
#include <ranges>
#include <span>
#include <vector>

#include "vulkan/vulkan.hpp"
#include "vulkan/vulkan_raii.hpp"

#define BAIL_ON_BAD_RESULT(result)                                                                                     \
    if ((result) != VK_SUCCESS)                                                                                        \
    {                                                                                                                  \
        fprintf(stderr, "Failure at %u %s\n", __LINE__, __FILE__);                                                     \
        exit(-1);                                                                                                      \
    }

namespace
{
std::pair<VkResult, std::optional<size_t>> getBestComputeQueue(const auto& physicalDevice)
{
    const auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

    // first try and find a queue that has just the compute bit set
    for (size_t i = 0; const auto& prop : queueFamilyProperties)
    {
        // mask out the sparse binding bit that we aren't caring about (yet!) and
        // the transfer bit
        const auto maskedFlags =
            (~(vk::QueueFlagBits::eTransfer | vk::QueueFlagBits::eSparseBinding) & prop.queueFlags);

        if (!(vk::QueueFlagBits::eGraphics & maskedFlags) && (vk::QueueFlagBits::eCompute & maskedFlags))
        {
            return {VK_SUCCESS, i};
        }
        ++i;
    }

    // lastly get any queue that'll work for us
    for (size_t i = 0; const auto& prop : queueFamilyProperties)
    {
        // mask out the sparse binding bit that we aren't caring about (yet!) and
        // the transfer bit
        const auto maskedFlags =
            (~(vk::QueueFlagBits::eTransfer | vk::QueueFlagBits::eSparseBinding) & prop.queueFlags);

        if (vk::QueueFlagBits::eCompute & maskedFlags)
        {
            return {VK_SUCCESS, i};
        }
        ++i;
    }

    return {VK_ERROR_INITIALIZATION_FAILED, {}};
}
} // namespace

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

    const std::vector<const char*> Layers = {"VK_LAYER_KHRONOS_validation"};
    const vk::InstanceCreateInfo instanceCreateInfo(vk::InstanceCreateFlags(), &applicationInfo, Layers.size(),
                                                    Layers.data());

    const vk::raii::Context context;
    const auto instance = vk::raii::Instance(context, instanceCreateInfo);

    const auto physicalDevices = instance.enumeratePhysicalDevices();

    for (auto& physDev : physicalDevices)
    {
        const auto [result, queueFamilyIndex] = getBestComputeQueue(physDev);
        if (!queueFamilyIndex)
        {
            BAIL_ON_BAD_RESULT(result);
            exit(1);
        }

        constexpr std::array queuePrioritory = {1.0f};
        const auto deviceQueueCreateInfo =
            vk::DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), *queueFamilyIndex, queuePrioritory);

        const std::array queueInfos = {deviceQueueCreateInfo};
        const auto deviceCreateInfo = vk::DeviceCreateInfo(vk::DeviceCreateFlags(), queueInfos);

        const auto device = vk::raii::Device(physDev, deviceCreateInfo);
        const auto props = physDev.getMemoryProperties();

        const int32_t bufferLength = 16384;
        const auto bufferSize = sizeof(bufferLength) * bufferLength;

        const auto memorySize = bufferSize * 2;

        const auto memoryTypeIndex = [&props]() -> std::optional<size_t> {
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

        std::cout << *memoryTypeIndex << "\n";

        const vk::MemoryAllocateInfo memoryAllocateInfo(memorySize, *memoryTypeIndex);

        const vk::raii::DeviceMemory memory(device, memoryAllocateInfo);

        {
            int32_t* payload = static_cast<int32_t*>(memory.mapMemory(0, memorySize));
            if (!payload)
            {
                BAIL_ON_BAD_RESULT(VK_ERROR_OUT_OF_HOST_MEMORY);
            }
            auto payloadSpan = std::span<int32_t>(payload, memorySize / sizeof(int32_t));
            std::ranges::for_each(payloadSpan, [](auto& elem) { elem = std::rand(); });
            std::ranges::for_each_n(payload, 10, [](const auto& elem) { std::cout << elem << "\n"; });
        }

        std::cout << to_string(memory.debugReportObjectType) << "\n";

        memory.unmapMemory();
        const std::array indices = {static_cast<uint32_t>(*queueFamilyIndex)};
        const auto bufferCreateInfo =
            vk::BufferCreateInfo(vk::BufferCreateFlags(), bufferSize, vk::BufferUsageFlagBits::eStorageBuffer,
                                 vk::SharingMode::eExclusive, indices);
        const auto in_buffer = vk::raii::Buffer(device, bufferCreateInfo);

        const auto out_buffer = vk::raii::Buffer(device, bufferCreateInfo);

        in_buffer.bindMemory(*memory, 0);

        out_buffer.bindMemory(*memory, bufferSize);

        constexpr auto spirv = makeSpirvCode(static_cast<uint32_t>(bufferSize));
        (void)spirv;
    }
    return 0;
}