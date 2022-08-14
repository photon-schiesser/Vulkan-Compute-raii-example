cmake_minimum_required(VERSION 3.8)

project("compute-pipeline")

find_package(Vulkan REQUIRED)

add_executable(example example.cpp)

target_link_libraries(example Vulkan::Vulkan)

if(MSVC)
  target_compile_options(example PRIVATE /W4 /WX)
else()
  target_compile_options(example PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()