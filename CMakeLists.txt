cmake_minimum_required(VERSION 3.8)

project("compute-pipeline")

set(CMAKE_CXX_STANDARD_REQUIRED 20)
set(CMAKE_CXX_STANDARD 20)

find_package(Vulkan REQUIRED)

add_executable(example example.cpp makeSpirvCode.cpp)

target_link_libraries(example Vulkan::Vulkan)

if(MSVC)
  target_compile_options(example PRIVATE /W4 /WX)
else()
  target_compile_options(example PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

target_precompile_headers(example PUBLIC /usr/include/vulkan/vulkan.hpp PUBLIC /usr/include/vulkan/vulkan_raii.hpp)
