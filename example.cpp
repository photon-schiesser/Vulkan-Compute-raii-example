#include <array>
#include <iostream>
#include <optional>
#include <vector>

#include "vulkan/vulkan.hpp"

#define BAIL_ON_BAD_RESULT(result)                                                                                     \
    if ((result) != VK_SUCCESS)                                                                                        \
    {                                                                                                                  \
        fprintf(stderr, "Failure at %u %s\n", __LINE__, __FILE__);                                                     \
        exit(-1);                                                                                                      \
    }

namespace
{
std::pair<VkResult, std::optional<size_t>> getBestComputeQueue(const vk::PhysicalDevice& physicalDevice)
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

    const auto instance = vk::createInstance(instanceCreateInfo);

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
        std::cout << &deviceQueueCreateInfo << "\n";
    }
    return 0;
}