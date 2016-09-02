cmake_minimum_required(VERSION 2.8)
PROJECT(FASTCV_API)

## Options
option(USE_CUDA "whether to build on CUDA device" OFF)
option(USE_OPENCL "whether to build on opencl device" OFF)
# Please note this: this option is only work for X86 platform.
# and if USE_CPP11 is set to OFF, we consider the compiler is too old and
# does not support FMA instructions. (eg. Centos 6.X series)
option(USE_CPP11 "whether to build use CPP11 flag" ON)
option(DEBUG "whether to build to debug" OFF)

option(BUILD_ANDROID "whether to build on android platform" OFF)
option(BUILD_IOS "whether to build on ios platform" OFF)
option(BUILD_IOS_SIM "whether to build on ios_sim platform" OFF)
option(BUILD_WINDOWS "whether to build on Windows" OFF)
option(BUILD_BENCHMARK "whether to benchmark" OFF)
option(BUILD_ARM "whether to build on ARM device" OFF)
option(BUILD_I386 "whether to build on I386 platform(only use to build i386 on linux/unix x86_64 platform, such as linux_x86_64 and mac_x86_64)" OFF)

if(USE_CUDA AND USE_OPENCL)
    message(FATAL_ERROR "USE_CUDA and USE_OPENCL can't be setted on both at the same time!")
endif()
if(BUILD_ANDROID AND BUILD_IOS)
    message(FATAL_ERROR "BUILD_ANDROID and BUILD_IOS can't be setted on both at the same time!")
endif()
if((USE_CUDA OR USE_OPENCL) AND (BUILD_ANDROID OR BUILD_IOS))
    message(FATAL_ERROR "Only one of USE_CUDA, USE_OPENCL, BUILD_ANDROID and BUILD_IOS cat be setted on!")
endif()

#########################################
# Directory variables
#########################################
get_filename_component(HOME_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
SET(POST_RELEASE_LIB "${CMAKE_CURRENT_SOURCE_DIR}/lib")
if (EXISTS ${POST_RELEASE_LIB})
    message(STATUS "post_release_lib: ${POST_RELEASE_LIB}")
else()
    message(FATAL_ERROR "/post_release_lib not found.")
endif()

if(USE_OPENCL)
	SET(BCL_BASE "${HOME_DIR}/bcl")
	if (EXISTS ${BCL_BASE})
    message(STATUS "BCL_BASE: ${BCL_BASE}")
	else()
    message(FATAL_ERROR "/bcl not found.")
	endif()

    set(UNIVERSAL_BASE "${HOME_DIR}/universal")
    if (EXISTS ${UNIVERSAL_BASE})
        message(STATUS "Found Universal: ${UNIVERSAL_BASE}")
    else()
        message(FATAL_ERROR "Universal not found.")
    endif()
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${UNIVERSAL_BASE}/cmake)
    message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})

endif()

## Platform setting
if (BUILD_ANDROID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__ANDROID__")
    if(ARM_ARCH MATCHES ARMV7)
        #set(ANDROID_TOOLCHAIN_ROOT /home/android-toolchain-armv7)
        set(ANDROID_TOOLCHAIN_ROOT /home/lijiabin/tools/my-android-toolchain-armv7)
        set(CMAKE_FIND_ROOT_PATH "${ANDROID_TOOLCHAIN_ROOT}/sysroot")
        set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
        set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
        set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
        set(CMAKE_C_COMPILER "${ANDROID_TOOLCHAIN_ROOT}/bin/arm-linux-androideabi-gcc")
        set(CMAKE_CXX_COMPILER "${ANDROID_TOOLCHAIN_ROOT}/bin/arm-linux-androideabi-g++")
        set(ARCH ARM)
        set(CMAKE_SYSTEM_PROCESSOR armv7)
    elseif(ARM_ARCH MATCHES ARM64)
        #set(ANDROID_TOOLCHAIN_ROOT /home/android-toolchain-arm64)
        set(ANDROID_TOOLCHAIN_ROOT /home/lijiabin/tools/my-android-toolchain-aarch64)
        set(CMAKE_FIND_ROOT_PATH "${ANDROID_TOOLCHAIN_ROOT}/sysroot")
        set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
        set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
        set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
        set(CMAKE_C_COMPILER "${ANDROID_TOOLCHAIN_ROOT}/bin/aarch64-linux-android-gcc")
        set(CMAKE_CXX_COMPILER "${ANDROID_TOOLCHAIN_ROOT}/bin/aarch64-linux-android-g++")
        set(ARCH ARM)
        set(CMAKE_SYSTEM_PROCESSOR aarch64)
    endif()
endif()
if (BUILD_IOS)
    set(CMAKE_OSX_SYSROOT /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS9.2.sdk)
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
    set(ARCH ARM)
    if(ARM_ARCH MATCHES ARMV7)
        set(CMAKE_SYSTEM_PROCESSOR armv7)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -arch armv7")
    elseif(ARM_ARCH MATCHES ARM64)
        set(CMAKE_SYSTEM_PROCESSOR aarch64)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -arch arm64")
    endif()
endif()
if (BUILD_IOS_SIM)
    set(CMAKE_OSX_SYSROOT /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator9.2.sdk)
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
    set(ARCH X86)
    if(BUILD_I386)
        set(CMAKE_SYSTEM_PROCESSOR i386)
    else()
        set(CMAKE_SYSTEM_PROCESSOR x86_64)
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mios-simulator-version-min=7.0")
endif()
if (BUILD_ARM)
    if(ARM_ARCH MATCHES ARMV7)
        set(ARMLINUX_TOOLCHAIN_ROOT /usr/bin)
        set(CMAKE_C_COMPILER "${ARMLINUX_TOOLCHAIN_ROOT}/arm-linux-gnueabihf-gcc")
        set(CMAKE_CXX_COMPILER "${ARMLINUX_TOOLCHAIN_ROOT}/arm-linux-gnueabihf-g++")
        set(CMAKE_SYSTEM_PROCESSOR armv7)
    elseif(ARM_ARCH MATCHES ARM64)
        set(ARMLINUX_TOOLCHAIN_ROOT /home/arm-toolchain-arm64/bin)
        set(CMAKE_C_COMPILER "${ARMLINUX_TOOLCHAIN_ROOT}/aarch64-linux-gnu-gcc")
        set(CMAKE_CXX_COMPILER "${ARMLINUX_TOOLCHAIN_ROOT}/aarch64-linux-gnu-g++")
        set(CMAKE_SYSTEM_PROCESSOR aarch64)
    endif()
    set(ARCH ARM)
endif()

if (BUILD_I386)
    set(IS_I386 "-m32")
endif()
if (DEBUG)
    set(CXX_FLAG_O "-O0 -g")
else()
    set(CXX_FLAG_O "-O3")
endif()

## Compiler setting
message(STATUS "CMAKE_SYSTEM_NAME: " ${CMAKE_SYSTEM_NAME})
message(STATUS "CXX_COMPILER_ID: " ${CMAKE_CXX_COMPILER_ID})
message(STATUS "CXX_COMPILER_VERSION: " ${CMAKE_CXX_COMPILER_VERSION})
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CXX_COMMON_FLAGS "${CXX_FLAG_O} -fPIC -Wall ${IS_I386}")
    if (USE_CPP11)
        add_definitions(-DUSE_CPP11)
        set(CPP11_ENABLED_FLAGS "-std=c++11")
    else()
        set(CPP11_ENABLED_FLAGS "-std=c++0x")
    endif()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CXX_COMMON_FLAGS "${CXX_FLAG_O} -fPIC ${IS_I386}")
    if (USE_CPP11)
        add_definitions(-DUSE_CPP11)
        set(CPP11_ENABLED_FLAGS "-std=c++11 -stdlib=libc++")
    else()
        set(CPP11_ENABLED_FLAGS "")
    endif()
    set(CMAKE_MACOSX_RPATH ON)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CPP11_ENABLED_FLAGS "")
    set(WARNING_ENABLED_FLAGS "-W -Wall")
    set(CXX_COMMON_FLAGS "/O2")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    set(CMAKE_IMPORT_LIBRARY_PREFIX "lib")
    set(CMAKE_SHARED_LIBRARY_PREFIX "lib")
    set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
    if(NOT BUILD_WINDOWS)
        set(BUILD_WINDOWS ON)
    endif()
else()
    message(FATAL_ERROR, "Error: Compiler is not supported yet.")
endif()
# enable asm
enable_language(ASM)

## Architecture setting
message(STATUS "CMAKE_SYSTEM_PROCESSOR: " ${CMAKE_SYSTEM_PROCESSOR})

# linux and mac os
if (CMAKE_SYSTEM_PROCESSOR MATCHES x86* OR CMAKE_SYSTEM_PROCESSOR MATCHES i.86 OR BUILD_I386)
    set(ARCH X86)
    set(AVX_ENABLED_FLAGS "-mavx")
    set(FMA_ENABLED_FLAGS "-mfma")
    set(CMAKE_CXX_FLAGS "${CXX_COMMON_FLAGS} ${CPP11_ENABLED_FLAGS}")
    # windows os
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES AMD64)
    set(ARCH X86)
    set(AVX_ENABLED_FLAGS "/arch:AVX")
    set(FMA_ENABLED_FLAGS "/arch:AVX2")
    set(CMAKE_CXX_FLAGS "${CXX_COMMON_FLAGS} ${CPP11_ENABLED_FLAGS}")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES arm*)
    set(ARCH ARM)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES armv7*)
        set(ARM_ARCH ARMV7)
        add_definitions(-DUSE_ARMV7)
        if(MFLOAT_ABI MATCHES hard)
            set(MFLOAT_ABI "hard")
        else()
            set(MFLOAT_ABI "softfp")
        endif()
        message(STATUS "ARM v7 processor")
        message(STATUS "ARM NEON Optimization is OPEN")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CPP11_ENABLED_FLAGS} ${CXX_COMMON_FLAGS} -mfloat-abi=${MFLOAT_ABI} -mfpu=neon -ftree-vectorize -flax-vector-conversions")
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfp16-format=ieee")
        endif()
        if (BUILD_IOS)
            set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -arch armv7 -mfloat-abi=${MFLOAT_ABI} -mfpu=neon")
        endif()
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES aarch64)
        set(ARM_ARCH ARM64)
        add_definitions(-DUSE_ARM64)
        message(STATUS "ARM 64 processor")
        message(STATUS "ARM NEON Optimization is OPEN")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CPP11_ENABLED_FLAGS} ${CXX_COMMON_FLAGS} -ftree-vectorize -flax-vector-conversions")
        if (BUILD_IOS)
            set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -arch arm64")
        endif()
    else()
        message(STATUS "Warning: Architecture is not armv7 or arm64 which have not been tested yet.")
    endif()
else()
    set(ARCH UNSUPPORTED)
    message(FATAL_ERROR "Error: Architecture is not supported.")
endif()

message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})



#########################################
## Dependencies
#########################################

## CUDA and OpenCL settings

if (USE_CUDA)
    find_package(CUDA "7.0")
    if (CUDA_FOUND)
        message(STATUS "CUDA_VERSION: " ${CUDA_VERSION})
        message(STATUS "CUDA_TOOLKIT_ROOT_DIR: " ${CUDA_TOOLKIT_ROOT_DIR})
        set(CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30
        -gencode arch=compute_35,code=sm_35
        -gencode arch=compute_37,code=sm_37
        -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_52,code=sm_52
        -gencode arch=compute_52,code=compute_52
        -Xcompiler -fPIC -Xcompiler -O3 -Xcompiler -Wall")
        set(CUDA_PROPAGATE_HOST_FLAGS OFF)
        add_definitions(-DUSE_CUDA)
        add_definitions(-DFASTCV_USE_CUDA)
        add_definitions(-DUSE_${ARCH})
        add_definitions(-DFASTCV_USE_${ARCH})
        set(GPUS CUDA)
        set(GPUS_LOWER cuda)
        include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    else()
        message(FATAL_ERROR "You set USE_CUDA ON, but CUDA package not found, please check your system environment, or set USE_CUDA OFF and retry.")
    endif()
endif()
if (USE_OPENCL)
    find_package(OpenCL)
    if (OpenCL_FOUND)
        message(STATUS "OpenCL_VERSION_MAJOR: " ${OpenCL_VERSION_MAJOR})
        message(STATUS "OpenCL_VERSION_MINOR: " ${OpenCL_VERSION_MINOR})
        message(STATUS "OpenCL_INCLUDE_DIRS: " ${OpenCL_INCLUDE_DIRS})
        message(STATUS "OpenCL_LIBRARIES: " ${OpenCL_LIBRARIES})
        add_definitions(-DUSE_OCL)
        add_definitions(-DFASTCV_USE_OCL)
        include_directories(SYSTEM ${OpenCL_INCLUDE_DIRS})

    else()
        message(FATAL_ERROR "You set USE_OPENCL ON, but OpenCL package not found, please check your system environment, or set USE_OPENCL OFF and retry.")
    endif()
else()
    add_definitions(-DUSE_${ARCH})
    add_definitions(-DFASTCV_USE_${ARCH})
endif()

## Gtest setting
if ((NOT BUILD_WINDOWS) AND (NOT BUILD_ANDROID) AND (NOT BUILD_IOS) AND (NOT BUILD_IOS_SIM) AND (NOT BUILD_ARM) AND (NOT BUILD_I386) AND (NOT CMAKE_HOST_SYSTEM_NAME MATCHES Darwin*))
    find_package(GTest REQUIRED)
    # Pthread library is required for GTest to work properly
    find_package(Threads REQUIRED)
endif()
if (GTEST_FOUND)
    include_directories(SYSTEM ${GTEST_INCLUDE_DIR})
    message(STATUS "GTEST_INCLUDE_DIR: ${GTEST_INCLUDE_DIR}")
endif()

## find cppcheck, to detect common syntax error in src
## notice: cppcheck is integrated into test
## usage: execute "make test" after "cmake ..", detail results in "./Testing/Temporary/LastTest.log"
find_package(cppcheck QUIET)
if(CPPCHECK_FOUND)
    if("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" GREATER 3.2)
        cmake_policy(SET CMP0040 OLD)
    endif()
    message(STATUS "Cppcheck Found")
    message(STATUS "CPPCHECK_EXECUTABLE: ${CPPCHECK_EXECUTABLE}")
    include(${CMAKE_MODULE_PATH}/CppcheckTargets.cmake)
else()
    message(STATUS "Cppcheck Not Found")
endif()


## img decode setting
find_package(JPEG)
if(JPEG_FOUND)
    include_directories(SYSTEM ${JPEG_INCLUDE_DIR})
    message(STATUS "JPEG_INCLUDE_DIR: ${JPEG_INCLUDE_DIR}")
    add_definitions(-DFASTCV_USE_JPEG)
endif()

find_package(PNG)
if(PNG_FOUND)
    include_directories(SYSTEM ${PNG_INCLUDE_DIR})
    message(STATUS "PNG_INCLUDE_DIR: ${PNG_INCLUDE_DIR}")
    add_definitions(-DFASTCV_USE_PNG)
endif()

SET(API_INC include)
SET(INSIDE_INC src/include)
SET(TESTS_DIR tests)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
MESSAGE(STATUS "fastcv_include_path:" ${FASTCV_INCLUDE_BASE})

## Include directory

if(USE_OPENCL)
    include_directories(SYSTEM ${FASTCV_INCLUDE_BASE} ${API_INC} ${INSIDE_INC} ${BCL_BASE} ${UNIVERSAL_BASE}/include)
else()
include_directories(SYSTEM ${FASTCV_INCLUDE_BASE} ${API_INC} ${INSIDE_INC})
endif()

## Add Library
## set fmath link lib name and fastcv output lib name
message(STATUS "CMAKE_HOST_SYSTEM_NAME: ${CMAKE_HOST_SYSTEM_NAME}")
set(TARGET_OS ${CMAKE_HOST_SYSTEM_NAME})
if(CMAKE_HOST_SYSTEM_NAME MATCHES Linux*)
    set(TARGET_OS linux)
elseif(CMAKE_HOST_SYSTEM_NAME MATCHES Darwin*)
    set(TARGET_OS mac)
elseif(CMAKE_HOST_SYSTEM_NAME MATCHES Windows*)
    set(TARGET_OS windows)
endif()

if(BUILD_IOS)
    set(TARGET_OS ios)
elseif(BUILD_IOS_SIM)
    set(TARGET_OS ios_sim)
elseif(BUILD_ANDROID)
    set(TARGET_OS android)
endif()
message(STATUS "CMAKE_TARGET_SYSTEM_NAME: ${TARGET_OS}")

if(CUDA_FOUND)
    set(LIB_NAME_CUDA_SUFFIX _cuda${CUDA_VERSION_MAJOR}_${CUDA_VERSION_MINOR})
endif()
if(OpenCL_FOUND)
    set(LIB_NAME_OCL_SUFFIX _opencl${OpenCL_VERSION_MAJOR}_${OpenCL_VERSION_MINOR})
endif()
message(STATUS "LIB_NAME_SUFFIX: ${LIB_NAME_CUDA_SUFFIX} ${LIB_NAME_OCL_SUFFIX}")

if(BUILD_WINDOWS)
    set(WIN_PREFIX lib)
    if(CMAKE_CL_64)
        set(PLATFORM_NAME x86_64)
    else()
        set(PLATFORM_NAME x86)
    endif()
    set(BUILDTYPE_SUFFIX "_$(ConfigurationName)")
else()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES i.86 OR BUILD_I386)
        set(PLATFORM_NAME x86)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES x86*)
        set(PLATFORM_NAME x86_64)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES armv7*)
        set(PLATFORM_NAME armv7)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES aarch64)
        set(PLATFORM_NAME arm64)
    endif()
endif()

# Link directory

if(CUDA_FOUND)
    link_directories(${LIBRARY_OUTPUT_PATH} ${POST_RELEASE_LIB}/fmath/${TARGET_OS}-${PLATFORM_NAME}${LIB_NAME_CUDA_SUFFIX}${BUILDTYPE_SUFFIX})
    link_directories(${LIBRARY_OUTPUT_PATH} ${POST_RELEASE_LIB}/fastcv/${TARGET_OS}-${PLATFORM_NAME}${LIB_NAME_CUDA_SUFFIX}${BUILDTYPE_SUFFIX})
    set(FMATH_LIB_CUDA_NAME ${WIN_PREFIX}fmath_${TARGET_OS}_${PLATFORM_NAME}${LIB_NAME_CUDA_SUFFIX})
    set(FASTCV_LIB_CUDA_NAME ${WIN_PREFIX}fastcv_${TARGET_OS}_${PLATFORM_NAME}${LIB_NAME_CUDA_SUFFIX})
    set(FMATH_LIB_NAME ${FMATH_LIB_NAME} ${FMATH_LIB_CUDA_NAME})
endif()
if(OpenCL_FOUND)
    link_directories(${LIBRARY_OUTPUT_PATH} ${POST_RELEASE_LIB}/fmath/${TARGET_OS}-${PLATFORM_NAME}${LIB_NAME_OCL_SUFFIX}${BUILDTYPE_SUFFIX})
    link_directories(${LIBRARY_OUTPUT_PATH} ${POST_RELEASE_LIB}/fastcv/${TARGET_OS}-${PLATFORM_NAME}${LIB_NAME_OCL_SUFFIX}${BUILDTYPE_SUFFIX})
    set(FMATH_LIB_OCL_NAME ${WIN_PREFIX}fmath_${TARGET_OS}_${PLATFORM_NAME}${LIB_NAME_OCL_SUFFIX})
    set(FASTCV_LIB_OCL_NAME ${WIN_PREFIX}fastcv_${TARGET_OS}_${PLATFORM_NAME}${LIB_NAME_OCL_SUFFIX})
    set(FMATH_LIB_NAME ${FMATH_LIB_NAME} ${FMATH_LIB_OCL_NAME})
endif()
if((ARCH MATCHES X86) OR (ARCH MATCHES ARM))
    link_directories(${LIBRARY_OUTPUT_PATH} ${POST_RELEASE_LIB}/fmath/${TARGET_OS}-${PLATFORM_NAME}${BUILDTYPE_SUFFIX})
    link_directories(${LIBRARY_OUTPUT_PATH} ${POST_RELEASE_LIB}/fastcv/${TARGET_OS}-${PLATFORM_NAME}${BUILDTYPE_SUFFIX})
    set(FMATH_LIB_${ARCH}_NAME ${WIN_PREFIX}fmath_${TARGET_OS}_${PLATFORM_NAME})
    set(FASTCV_LIB_${ARCH}_NAME ${WIN_PREFIX}fastcv_${TARGET_OS}_${PLATFORM_NAME})
    set(FMATH_LIB_NAME ${FMATH_LIB_NAME} ${FMATH_LIB_${ARCH}_NAME})
endif()

### set link libs
set(FASTCV_ARCH_LINKLIB
    ${BLAS_LIBRARIES}
    )
if(JPEG_FOUND)
    set(FASTCV_ARCH_LINKLIB ${FASTCV_ARCH_LINKLIB} ${JPEG_LIBRARIES})
endif()
if(PNG_FOUND)
    set(FASTCV_ARCH_LINKLIB ${FASTCV_ARCH_LINKLIB} ${PNG_LIBRARIES})
endif()

set(FASTCV_CUDA_LINKLIB
    ${FASTCV_ARCH_LINKLIB}
    ${CUDA_CUBLAS_LIBRARIES}
    )

set(FASTCV_OPENCL_LINKLIB
    ${FASTCV_ARCH_LINKLIB}
    ${OpenCL_LIBRARIES}
    )

#########################################
# api library
#########################################
set(API_LINK_LIBS
    ${FASTCV_LIB_${ARCH}_NAME}
    ${FMATH_LIB_NAME})
if(CUDA_FOUND)
    set(API_LINK_LIBS
        ${API_LINK_LIBS}
        ${FASTCV_CUDA_LINKLIB}
        )
elseif(OpenCL_FOUND)
    set(API_LINK_LIBS
        ${API_LINK_LIBS}
        ${FASTCV_OPENCL_LINKLIB}
        )
else()
    set(API_LINK_LIBS
        ${API_LINK_LIBS}
        ${FASTCV_ARCH_LINKLIB}
        )
endif()
set(API_SRC_LIST
    src/convertTo.cpp
    src/crop.cpp
    src/cvtColor.cpp
    src/filter.cpp
    src/gaussianBlur.cpp
    src/imageReadWrite.cpp
    src/oclib.cpp
    src/resize.cpp
    src/rotate.cpp
    src/setTo.cpp
    src/splitMerge.cpp
    src/warpAffine.cpp
    src/arithmetic.cpp
    src/boxFilter.cpp
    src/integral.cpp
    src/fastBilateralFilter.cpp
    src/bilateralFilter.cpp
    #src/func_fastGuidedFilterReflect101_SameMat.cpp
    src/fastGuidedFilter.cpp
    src/opencl.cpp
    src/remap.cpp
    src/erode.cpp
    src/dilate.cpp
    )

if(OpenCL_FOUND)
    set(API_SRC_LIST
        ${API_SRC_LIST}
        src/opencl.cpp
        )
endif()

MESSAGE(STATUS "API_LINK_LIBS: " ${API_LINK_LIBS})

add_library(api_fastcv SHARED ${API_SRC_LIST})
set_target_properties(api_fastcv PROPERTIES CLEAN_DIRECT_OUTPUT 1)
target_link_libraries(api_fastcv ${API_LINK_LIBS})

add_library(api_fastcv_static STATIC ${API_SRC_LIST})
set_target_properties(api_fastcv_static PROPERTIES OUTPUT_NAME "api_fastcv")
set_target_properties(api_fastcv_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
target_link_libraries(api_fastcv_static ${API_LINK_LIBS})

if(CPPCHECK_FOUND)
    add_cppcheck_sources(api_fastcv ${API_SRC_LIST})
endif()

#########################################
# Unit Tests
#########################################

if (GTEST_FOUND)
    set(TESTS_LINK_LIBS
        ${TESTS_LINK_LIBS}
        ${GTEST_BOTH_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        ${CUDA_CUBLAS_LIBRARIES}
        api_fastcv
        ${API_LINK_LIBS}
        ${FASTCV_ARCH_LINKLIB}
        opencv_core
        opencv_imgproc
        opencv_highgui
        opencv_gpu
        opencv_ocl
        )
    if(CUDA_FOUND)
        set(TESTS_LINK_LIBS ${TESTS_LINK_LIBS} ${FASTCV_LIB_CUDA_NAME} ${CUDA_LIBRARIES})
    endif()
    if(OpenCL_FOUND)
        set(TESTS_LINK_LIBS ${TESTS_LINK_LIBS} ${FASTCV_LIB_OCL_NAME} ${OpenCL_LIBRARIES})
    endif()
    message(STATUS "TESTS_LINK_LIBS: ${TESTS_LINK_LIBS}")

    # FASTCV base gtest
    add_subdirectory(${TESTS_DIR})
endif()

## Add gtest
enable_testing()
