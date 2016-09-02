#ifndef HPC_UNIVERSAL_TYPES_H_
#define HPC_UNIVERSAL_TYPES_H_ 

#ifndef __CUDACC__
#define CUDAC inline
#else
#define CUDAC inline __host__ __device__
#endif

#ifdef __cplusplus
extern "C" {
#endif
typedef unsigned char uchar;
typedef char            int8;
typedef unsigned char   uint8;
typedef short           int16;
typedef unsigned short  uint16;
typedef unsigned short  ushort;
typedef int             int32;
typedef unsigned int    uint32;
typedef unsigned int    uint;
typedef float           float32;
typedef double          float64;

#ifdef __cplusplus
}
#endif

template<typename T>
struct Complex {
    T real;
    T image;
};
typedef Complex<float32> complex64;
typedef Complex<float64> complex128;

template<typename T>
struct V2 {
    T x;
    T y;
};
typedef V2<int8>        int8x2;
typedef V2<uint8>       uint8x2;
typedef V2<int16>       int16x2;
typedef V2<uint16>      uint16x2;
typedef V2<int32>       int32x2;
typedef V2<uint32>      uint32x2;
typedef V2<float32>     float32x2;
typedef V2<float64>     float64x2;
typedef V2<complex64>   complex64x2;

template<typename T>
struct V3 {
    T x;
    T y;
    T z;
};
typedef V3<int8>        int8x3;
typedef V3<uint8>       uint8x3;
typedef V3<int16>       int16x3;
typedef V3<uint16>      uint16x3;
typedef V3<int32>       int32x3;
typedef V3<uint32>      uint32x3;
typedef V3<float32>     float32x3;
typedef V3<float64>     float64x3;

template<typename T>
struct V4 {
    T x;
    T y;
    T z;
    T w;
};
typedef V4<int8>        int8x4;
typedef V4<uint8>       uint8x4;
typedef V4<int16>       int16x4;
typedef V4<uint16>      uint16x4;
typedef V4<int32>       int32x4;
typedef V4<uint32>      uint32x4;
typedef V4<float32>     float32x4;
typedef V4<float64>     float64x4;

#endif
