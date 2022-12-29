#include "gpuCopy.h"

#include <chrono>
#include <iostream>

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

    [[maybe_unused]] const auto physicalDevices = instance.enumeratePhysicalDevices();

    for (const auto& physDev : physicalDevices)
    {
        copyUsingDevice(physDev, bufferLength);
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