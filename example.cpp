#include "gpuCopy.h"

#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <ranges>
#include <span>
#include <vector>

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

int copyTest()
{
    constexpr vk::ApplicationInfo applicationInfo = []() {
        vk::ApplicationInfo temp;
        temp.pApplicationName = "Compute-Pipeline";
        temp.applicationVersion = 1;
        temp.pEngineName = nullptr;
        temp.engineVersion = 0;
        temp.apiVersion = VK_MAKE_VERSION(1, 1, 0);
        return temp;
    }();

    constexpr std::array layers = {"VK_LAYER_KHRONOS_validation"};
    const vk::InstanceCreateInfo instanceCreateInfo(vk::InstanceCreateFlags(), &applicationInfo,
                                                    layers.size(), layers.data());

    const vk::raii::Context context;
    const auto instance = vk::raii::Instance(context, instanceCreateInfo);

    constexpr uint32_t bufferLength = 16384 * 2 * 16 * 2;
    using bufferData_t = int32_t;
    constexpr uint32_t bufferSize = sizeof(bufferData_t) * bufferLength;
    [[maybe_unused]] constexpr auto memorySize = bufferSize * 2;

    [[maybe_unused]] const auto spirvFromFile = getSpirvFromFile("copy.comp.spv");

    [[maybe_unused]] const auto physicalDevices = instance.enumeratePhysicalDevices();

    for (const auto& physDev : physicalDevices)
    {
        copyUsingDevice(physDev);
    }
    return 0;
}

int main()
{
    auto clock = std::chrono::high_resolution_clock();
    const auto start = clock.now();
    copyTest();
    const auto stop = clock.now();

    std::cout << "Duration: " << std::chrono::duration<double, std::milli>(stop - start).count()
              << "\n";
    return 0;
}