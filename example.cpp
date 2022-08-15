// This is free and unencumbered software released into the public domain.
//
// Anyone is free to copy, modify, publish, use, compile, sell, or
// distribute this software, either in source code form or as a compiled
// binary, for any purpose, commercial or non-commercial, and by any
// means.
//
// In jurisdictions that recognize copyright laws, the author or authors
// of this software dedicate any and all copyright interest in the
// software to the public domain. We make this dedication for the benefit
// of the public at large and to the detriment of our heirs and
// successors. We intend this dedication to be an overt act of
// relinquishment in perpetuity of all present and future rights to this
// software under copyright law.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// For more information, please refer to <http://unlicense.org/>

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
        BAIL_ON_BAD_RESULT(result);

        // const float queuePrioritory = 1.0f;
        const auto deviceQueueCreateInfo =
            vk::DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), *queueFamilyIndex, {});
        std::cout << &deviceQueueCreateInfo << "\n";
    }
    return 0;
}