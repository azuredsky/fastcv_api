#ifndef HPC_UNIVERSAL_SYS_H_
#define HPC_UNIVERSAL_SYS_H_

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Programming ECO System
 **/
typedef enum {
    ECO_ENV_X86  = 1,
    ECO_ENV_ARM  = 2,
    ECO_ENV_CUDA = 4,
    ECO_ENV_OCL  = 8,
} EcoEnv_t;

/**
 * @biref System Platform
 */
typedef enum {
    PLATFORM_TYPE_INTEL     = 0,
    PLATFORM_TYPE_AMD       = 1,
    PLATFORM_TYPE_NVIDIA    = 2,
    PLATFORM_TYPE_ARM       = 3,
    PLATFORM_TYPE_QUALCOMM  = 4,
} PlatformType_t;


/**
 * @brief Instruction Set Architecture (ISA)
 *
 */
typedef enum {
    ISA_NOT_SUPPORTED       = -1,
#if defined(FASTCV_USE_X86)
    ISA_X86_AVX             = 0,
    ISA_X86_FMA             = 1, //contains AVX2
    ISA_X86_AVX512          = 2,
#endif
#if defined(FASTCV_USE_ARM)
    ISA_ARM_V7              = 101,
    ISA_ARM_V8              = 102,
#endif
#if defined(FASTCV_USE_CUDA)
    ISA_NV_KEPLER           = 201,
    ISA_NV_MAXWELL          = 202,
#endif
#if defined(FASTCV_USE_OCL)
    ISA_AMD_GCN             = 301,
    ISA_ARM_MALI            = 302,
#endif
} ISA_t;


/**
 * @brief isISA
 */
#if defined(FASTCV_USE_X86)
bool inline uniIsX86ISA(ISA_t isa) {
    return 0 == isa/100;
}
#endif
#if defined(FASTCV_USE_ARM)
bool inline uniIsARMISA(ISA_t isa) {
    return 1 == isa/100;
}
#endif
#if defined(FASTCV_USE_CUDA)
bool inline uniIsCUDAISA(ISA_t isa) {
    return 2 == isa/100;
}
#endif
#if defined(FASTCV_USE_OCL)
bool inline uniIsOCLISA(ISA_t isa) {
    return 3 == isa/100;
}
#endif


/**
 * @brief Implementation (IMP)
 */
typedef enum {
    IMP_NOT_SUPPORTED    = -1,
#if defined(FASTCV_USE_X86)
    IMP_X86_SANDYBRIDGE  = 0,
    IMP_X86_IVYBRIDGE    = 1,
    IMP_X86_HASWELL      = 12,
    IMP_X86_BROADWELL    = 13,
    IMP_X86_SKYLAKE      = 23,
#endif
#if defined(FASTCV_USE_ARM)
    IMP_ARM_A9           = 110,
    IMP_ARM_A15          = 111,
    IMP_ARM_A53          = 120,
    IMP_ARM_A57          = 121,
#endif
#if defined(FASTCV_USE_CUDA)
    IMP_NV_SM30          = 210,
    IMP_NV_SM32          = 211,
    IMP_NV_SM35          = 212,
    IMP_NV_SM50          = 220,
    IMP_NV_SM52          = 221,
#endif
#if defined(FASTCV_USE_OCL)
    IMP_AMD_GCN10        = 310,
    IMP_AMD_GCN11        = 311,
    IMP_AMD_GCN12        = 312,
#endif
} IMP_t;


/**
 * @brief check one implementaion is belong to isa
 */
bool inline uniIMPBelongToISA(ISA_t isa, IMP_t imp) {
    bool flag = ((isa/100) == (imp/100)); //same ISA
    flag = flag && ((isa%10) == (imp%100/10));
    return flag;
}

/**
 * @brief Properties
 */
#if defined(FASTCV_USE_X86)
typedef struct {
    char name[256];
    ISA_t isa;
    IMP_t imp;

    int numFlopsPerClockPerCore;
    float baseFrequency;// GHz
    float maxFrequency;// GHz

    size_t l1CacheSize;
    size_t l1CacheLineSize;
    size_t l2CacheSize;
    size_t l2CacheLineSize;
    size_t l3CacheSize;
    size_t l3CacheLineSize;
} X86Properties;
bool uniX86GetProperties(int core, X86Properties *p);
bool uniX86GetFlops(const X86Properties *p, float *baseGflops, float *maxGflops);
#endif
//comment this because we can't find a way to get all of them
#if defined(FASTCV_USE_ARM)
typedef struct {
    char name[256];
    ISA_t isa;
    IMP_t imp;

    int numFlopsPerClockPerCore;
    float baseFrequency;// GHz
    float maxFrequency;// GHz

    size_t l1CacheSize;
    size_t l1CacheLineSize;
    size_t l2CacheSize;
    size_t l2CacheLineSize;
} ARMProperties;
bool uniARMGetProperties(int core, ARMProperties *p);
bool uniARMGetFlops(const ARMProperties *p, float *baseGflops, float *maxGflops);
#endif
#if defined(FASTCV_USE_CUDA)
typedef struct {
    char name[256];
    ISA_t isa;
    IMP_t imp;

    int cc; //compute capability, 4 digits, 3050 for 3.5;5020 for 5.2
    int numCores; //num of compute cores
    int numFlopsPerClockPerCore;
    float baseFrequency;// GHz
    float maxFrequency;// GHz

    size_t sharedSizeInBytes;
    size_t totalDramSize;
} CUDAProperties;
bool uniCUDAGetProperties(int device, CUDAProperties *p);
bool uniCUDAGetFlops(const CUDAProperties *p, float *baseGflops, float* maxGflops);
#endif

#if defined(FASTCV_USE_OCL)
typedef struct {
    char name[256];
    ISA_t isa;
    IMP_t imp;

    int cc; //compute capability, 4 digits, 3050 for 3.5;5020 for 5.2
    int numCores; //num of compute cores
    int numFlopsPerClockPerCore;
    float baseFrequency;// GHz
    float maxFrequency;// GHz

    size_t sharedSizeInBytes;
    size_t totalDramSize;
} OCLProperties;
bool uniOCLGetProperties(int device, OCLProperties *p);
bool uniOCLGetFlops(const OCLProperties *p, float *baseGflops, float* maxGflops);
#endif


/**
 * @brief bind current thread to one processor
 *
 * @param coreId          processor ID for current thread to bind
 */
#if defined(FASTCV_USE_X86)
bool uniX86BindThreadToCore(int coreId);
#endif
#if defined(FASTCV_USE_ARM)
bool uniARMBindThreadToCore(int coreId);
#endif

#ifndef HPC_ALIGNMENT
#define HPC_ALIGNMENT 64
#endif

/**
 * @brief Malloc and free data
 */
#if defined(FASTCV_USE_X86)
inline bool x86Malloc(void** d, size_t size) {
    if(size % HPC_ALIGNMENT != 0) size = size - size%HPC_ALIGNMENT + HPC_ALIGNMENT;
#ifdef _WIN64
    *d = _aligned_malloc(size, HPC_ALIGNMENT);
#elif __linux__
    *d = aligned_alloc(HPC_ALIGNMENT, size);
#else
    *d = malloc(size);
#endif
    if(NULL == *d) return false;
    return true;
}

inline bool x86Free(void* d) {
    if(NULL == d) return false;
#if defined(_WIN64) || defined(_WIN32)
    _aligned_free(d);
#elif __linux__
    free(d);
#else
    free(d);
#endif
    return true;
}
#endif

#if defined(FASTCV_USE_ARM)
inline bool armMalloc(void** d, size_t size) {
    if(size % HPC_ALIGNMENT != 0) size = size - size%HPC_ALIGNMENT + HPC_ALIGNMENT;
#ifdef __ANDROID__
    *d = memalign(HPC_ALIGNMENT, size);
#elif __APPLE__
    *d = malloc(size);
#else
    *d = aligned_alloc(HPC_ALIGNMENT, size);
#endif
    if(NULL == *d) return false;
    return true;
}

inline bool armFree(void* d) {
    if(NULL == d) return false;
    free(d);
    return true;
}
#endif

#ifdef __cplusplus
}
#endif

#endif
