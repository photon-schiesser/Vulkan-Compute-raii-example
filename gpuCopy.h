#define VULKAN_HPP_NO_SMART_HANDLE
#include "vulkan/vulkan.hpp"
#include "vulkan/vulkan_raii.hpp"

int copyUsingDevice(const vk::raii::PhysicalDevice& physDev, uint32_t bufferLength);
