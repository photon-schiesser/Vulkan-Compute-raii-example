cmake_minimum_required(VERSION 3.8)

project("compute-pipeline")

set(CMAKE_CXX_STANDARD_REQUIRED 20)
set(CMAKE_CXX_STANDARD 20)

find_package(Vulkan REQUIRED)

add_executable(example example.cpp makeSpirvCode.cpp)

target_link_libraries(example PRIVATE Vulkan::Vulkan)

if(MSVC)
  target_compile_options(example PRIVATE /W4 /WX)
else()
  target_compile_options(example PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(Vulkan_glslangValidator_FOUND)
  message("Found glslang")
  else()
  message("NOT FOUND")
endif()

add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/copy.comp.spv"
    COMMAND glslangValidator -H -V -o "${CMAKE_BINARY_DIR}/copy.comp.spv" "copy.comp"
    DEPENDS "copy.comp"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Buiding Shaders"
)
add_custom_target(ComputeShader DEPENDS "${CMAKE_BINARY_DIR}/copy.comp.spv")
add_dependencies(example ComputeShader)

target_precompile_headers(example PUBLIC /usr/include/vulkan/vulkan.hpp PUBLIC /usr/include/vulkan/vulkan_raii.hpp)
