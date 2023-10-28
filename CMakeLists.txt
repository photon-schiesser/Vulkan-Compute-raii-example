cmake_minimum_required(VERSION 3.7)

project("compute-pipeline")

set(CMAKE_CXX_STANDARD_REQUIRED 23)
set(CMAKE_CXX_STANDARD 23)

#include the vulkan sdk headers
#include_directories("/home/user/vulkanSDK-1.3.239.0/x86_64/include/")

find_package(Vulkan REQUIRED)

include_directories( ${Vulkan_INCLUDE_DIRS} )

add_executable(example example.cpp makeSpirvCode.cpp gpuCopy.cpp)

target_link_libraries(example PRIVATE Vulkan::Vulkan)

message(STATUS "${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}")

if(MSVC)
  target_compile_options(example PRIVATE /W4 /WX)
else()
  target_compile_options(example PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/copy.comp.spv"
    COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -H -V -o "${CMAKE_BINARY_DIR}/copy.comp.spv" "copy.comp"
    DEPENDS "copy.comp"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Shaders"
)
add_custom_target(ComputeShader DEPENDS "${CMAKE_BINARY_DIR}/copy.comp.spv")
add_dependencies(example ComputeShader)

target_precompile_headers(example PUBLIC ${Vulkan_INCLUDE_DIRS}/vulkan/vulkan.hpp PUBLIC ${Vulkan_INCLUDE_DIRS}/vulkan/vulkan_raii.hpp)
