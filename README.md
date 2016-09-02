# fastcv_api : *fastcv api*

*fastcv_api*, is API of fastcv, it make fastcv to use some platforms, such as X86, ARM, NVIDIA GPU, OpenCL and so on. Then you can use X86/ARM and CUDA together. OCL doesnot provide temporarily.

## Prerequisites

fastcv has several dependencies.

- **CUDA** is required for GPU CUDA mode. (>=7.0)
- **OPENCL** is required for OpenCL mode.
- **gtest** is required for unit test. (>=1.7)
- **OpenCV** is required for unit test. (>=2)
- **CMake** is required for compilation. (>=2.8)

## Support

Until now, fastcv_api has provided many basic image operations, which mainly contains image resize, convertTO, warpAffine, serveral filters, gaussianBlur, integral, crop, cvtColor, etc.

The library is still in updating. We hope our fastcv_api is strong enough to replace opencv.


## Installation

- **post_release_lib**: The dependant project named with post_release_lib is needed, which provides basic libiary.
    And post_release_lib should be installed in the same directory with fastcv_api project.
    if link library is failed, please check the library whether in this project.
    git@gitlab.sz.sensetime.com:lijiabin/post_release_lib.git
- **bcl**: The dependant project named with bcl is needed when you build fastcv_api in opencl mode.
    And bcl should be installed in the same directory with fastcv_api project.

### Complilation

#### Options
- **USE_CUDA**   : whether to build on CUDA and x86/ARM device. (default: OFF)
- **USE_OPENCL** : whether to build on opencl and x86/ARM device. (default: OFF)
- **USE_CPP11**  : whether to open C++11 supported. (default: ON)

- **BUILD_ARM**     : whether to build on ARM platform. (default: OFF)
- **BUILD_ANDROID** : whether to build on android platform. (default: OFF)
- **BUILD_IOS**     : whether to build on ios platform. (default: OFF)
- **BUILD_IOS_SIM** : whether to build on ios sim platform. (default: OFF)
- **BUILD_WINDOWS** : whether to build on Windows. (default: OFF)
- **ARM_ARCH**      : specify ARM architecture. (accepted value: ARMV7 or ARM64)

#### Notice:
    - USE_CUDA and USE_OPENCL can't be setted on both at the same time.
    - BUILD_ANDOIRD and BUILD_IOS can't be setted on both at the same time.
    - Only one of USE_CUDA, USE_OPENCL, BUILD_ANDROID, BUILD_IOS can be setted on at the same time.

**Linux and Mac System :**
- cd fastcv_api && mkdir build && cd build
- cmake ..
- make

**Android**
Please modify `CMakeList.txt` to set variable `ANDROID_TOOLCHAIN_ROOT` which is android cross
compilation tools directory.
- cd fastcv_api && mkdir build && cd build
- cmake -D BUILD_ANDROID=ON -D ARM_ARCH=ARMV7|ARM64 ..
- make

**IOS**
Please modify `CMakeList.txt` to set variable `CMAKE_OSX_SYSROOT` which is ios sdk directory.
- cd fastcv_api && mkdir build && cd build
- cmake -D BUILD_IOS=ON -D ARM_ARCH=ARMV7|ARM64 ..
- make
