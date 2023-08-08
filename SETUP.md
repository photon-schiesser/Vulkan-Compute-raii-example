# Setup

Set up your environment to leverage the Vulkan SDK to execute GPU-advantage tasks.

This guide uses a local install of the Vulkan SDK, but can be configured to use an installation provided by your system package manager

Heavily borrowed from https://vulkan.lunarg.com/doc/view/latest/linux/getting_started.html

Tested with:
* Debian Bookworm 
* NVIDIA Driver version 470.182.03
* GTX 750
* Vulkan SDK 1.3.239.0

## Packages
Install dependencies of the Vulkan SDK and this project
```
sudo apt install vulkan-tools nvidia-vulkan-common libvulkan1 libvulkan-dev libvkfft-dev qtbase5-dev libxcb-xinput0 libxcb-xinerama0 cmake git make
```

## Vulkan SDK
1. Download and extract the Vulkan SDK
1. Source the environment file from the Vulkan SDK
```
cd vulkanSDK-1.3.239.0
source setup-env.sh
cd
```

## Project Setup
Clone the project and build it with CMake. Use the `CMAKE_INCLUDE_PATH` flag to specify the path to the extracted VulkanSDK on your file system
```
cd Vulkan-Compute-raii-example
mkdir cmake-build
cd cmake-build
cmake .. -DCMAKE_INCLUDE_PATH="/home/user/vulkanSDK-1.3.239.0/x86_64/"
make -j4
```

## Run the Examples
A successful build will produce a binary `examples` in the `cmake-build` directory. 
Run the produced binary via `./examples`. 
If your environment is correctly setup, you should see messages indicating GPU execution:
```
./examples
...
Duration of copying data 10 times on GPU: 10.7508
...
Duration of copying data 10 times on GPU: 9.54121
...
Duration of copying data 10 times on GPU: 268.537
```


If a message like this appears when running the `examples` binary:
```
terminate called after throwing an instance of 'vk::LayerNotPresentError'
  what():  vkCreateInstance: ErrorLayerNotPresent
```
Double check that your environment has properly sourced the `setup-env.sh` file in your Vulkan SDK install. 

Executing `env | grep -i vul` should produce entries that correctly map to resources in the Vulkan SDK.
