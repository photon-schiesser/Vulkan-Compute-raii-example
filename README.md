# Vulkan-Compute-raii-example
A simple example of setting up a vulkan compute pipeline using the vk::raii namespace of the Vulkan SDK.
This was created as a learning exercise.

## Inspiration
I was inspired by this simple example using C: https://gist.github.com/sheredom/523f02bbad2ae397d7ed255f3f3b5a7f

I started out by converting that example to use the Vulkan C++ RAII interface provided by vulkan_raii.hpp. Why? To learn how to set up a Vulkan compute pipeline. I also wanted to use the RAII interface specifically to avoid managing lifetimes (As it turns out, there is still some complexity with the lifetimes of objects that an application must satisfy according to the Vulkan spec, some more obvious than others, that must be managed by the programmer).

## Additional changes
The first thing I had to do was to set up a build system. I chose cmake and figured out how to add the Vulkan SDK library.

After converting the example to C++, I wanted to understand what the shader code was doing. I actually decompiled the SPIR-V code in the makeSpirvCode.h file to GLSL. This helped me understand how certain parts of the C++ code were mirroring the shader code (like the buffer descriptors).

I wanted to change the shader code, so I added the capability to compile the compute shader to the cmake file. 

The first customization I made to the shader was to allow the application to specify the group size in the shader. This required binding specialization constants in the pipeline on the C++ side. 

## Setup
[Setup](SETUP.md) - Follow this guide to set up your environment and run the example program.
