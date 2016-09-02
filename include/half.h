
#ifndef HALF_H_
#define HALF_H_

#if defined(FASTCV_USE_CUDA) || defined(FASTCV_USE_OCL)
#include <cmath>
#include <cstdio>
#include <cfloat>
#include <climits>

#ifdef _MSC_VER
typedef __int16 int16_t;
#else
#include <inttypes.h>
#endif

#if (CUDA_VERSION >= 7050 && defined(__CUDA_ARCH__))
  #define CUDA_HALF 1
  #include <cuda_fp16.h>
  /*! \brief __half2float_warp */
  __host__ __device__ float __half2float_warp(const volatile __half& h) { /* NOLINT(*) */
    __half val;
    val.x = h.x;
    return __half2float(val);
  }
#else
  #define CUDA_HALF 0
#endif

#ifndef __CUDACC__
#define CUDAC inline
#else
#define CUDAC inline __host__ __device__
#endif

/* \brief name space for host/device portable half-precision floats */
namespace uni {
namespace half {
#define HALF_OPERATOR(RTYPE, OP)                                  \
  CUDAC RTYPE operator OP (half_t a, half_t b) {                \
    return RTYPE(float(a) OP float(b));  /* NOLINT(*) */                  \
  }                                                                       \
  template<typename T>                                                    \
  CUDAC RTYPE operator OP (half_t a, T b) {                     \
    return RTYPE(float(a) OP float(b));  /* NOLINT(*) */                  \
  }                                                                       \
  template<typename T>                                                    \
  CUDAC RTYPE operator OP (T a, half_t b) {                     \
    return RTYPE(float(a) OP float(b));  /* NOLINT(*) */                  \
  }

#define HALF_ASSIGNOP(AOP, OP)                                    \
  template<typename T>                                                    \
  CUDAC half_t operator AOP (const T& a) {                      \
    return *this = half_t(float(*this) OP float(a));  /* NOLINT(*)*/      \
  }                                                                       \
  template<typename T>                                                    \
  CUDAC half_t operator AOP (const volatile T& a) volatile {    \
    return *this = half_t(float(*this) OP float(a));  /* NOLINT(*)*/      \
  }

#if CUDA_HALF
#define HALF_CONVERSIONOP(T)                                      \
  CUDAC operator T() const {                                    \
    return T(__half2float(cuhalf_));  /* NOLINT(*)*/                      \
  }                                                                       \
  CUDAC operator T() const volatile {                           \
    return T(__half2float_warp(cuhalf_));  /* NOLINT(*)*/                      \
  }
#else
#define HALF_CONVERSIONOP(T)                                      \
  CUDAC operator T() const {                                    \
    return T(half2float(half_));  /* NOLINT(*)*/                          \
  }                                                                       \
  CUDAC operator T() const volatile {                           \
    return T(half2float(half_));  /* NOLINT(*)*/                          \
  }
#endif  // CUDA_HALF

class half_t {
 public:
  union {
    uint16_t half_;
#if CUDA_HALF
    __half cuhalf_;
#endif  // CUDA_HALF
  };

  static CUDAC half_t Binary(uint16_t value) {
    half_t res;
    res.half_ = value;
    return res;
  }

  CUDAC half_t() {}

#if CUDA_HALF
  CUDAC explicit half_t(const __half& value) {
    cuhalf_ = value;
  }
#endif  // CUDA_HALF

  CUDAC half_t(const float& value) { constructor(value); }
  CUDAC explicit half_t(const double& value) { constructor(value); }
  CUDAC explicit half_t(const uint8_t& value) { constructor(value); }
  CUDAC explicit half_t(const int32_t& value) { constructor(value); }

  HALF_CONVERSIONOP(float)

  HALF_ASSIGNOP(+=, +)
  HALF_ASSIGNOP(-=, -)
  HALF_ASSIGNOP(*=, *)
  HALF_ASSIGNOP(/=, /)

  CUDAC half_t operator+() {
    return *this;
  }

  CUDAC half_t operator-() {
    return half_t(-float(*this));  // NOLINT(*)
  }

  CUDAC half_t operator=(const half_t& a) {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  CUDAC half_t operator=(const T& a) {
    return *this = half_t(a);  /* NOLINT(*)*/
  }

  CUDAC half_t operator=(const half_t& a) volatile {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  CUDAC half_t operator=(const T& a) volatile {
    return *this = half_t(a);  /* NOLINT(*)*/
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static int const shift = 13;
  static int const shiftSign = 16;

  static int32_t const infN = 0x7F800000;  // flt32 infinity
  static int32_t const maxN = 0x477FE000;  // max flt16 normal as a flt32
  static int32_t const minN = 0x38800000;  // min flt16 normal as a flt32
  static int32_t const signN = 0x80000000;  // flt32 sign bit

  static int32_t const infC = infN >> shift;
  static int32_t const nanN = (infC + 1) << shift;  // minimum flt16 nan as a flt32
  static int32_t const maxC = maxN >> shift;
  static int32_t const minC = minN >> shift;
  static int32_t const signC = signN >> shiftSign;  // flt16 sign bit

  static int32_t const mulN = 0x52000000;  // (1 << 23) / minN
  static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))

  static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted
  static int32_t const norC = 0x00400;  // min flt32 normal down shifted

  static int32_t const maxD = infC - maxC - 1;
  static int32_t const minD = minC - subC - 1;

  CUDAC uint16_t float2half(const float& value) const {
    Bits v, s;
    v.f = value;
    uint32_t sign = v.si & signN;
    v.si ^= sign;
    sign >>= shiftSign;  // logical shift
    s.si = mulN;
    s.si = s.f * v.f;  // correct subnormals
    v.si ^= (s.si ^ v.si) & -(minN > v.si);
    v.si ^= (infN ^ v.si) & -((infN > v.si) & (v.si > maxN));
    v.si ^= (nanN ^ v.si) & -((nanN > v.si) & (v.si > infN));
    v.ui >>= shift;  // logical shift
    v.si ^= ((v.si - maxD) ^ v.si) & -(v.si > maxC);
    v.si ^= ((v.si - minD) ^ v.si) & -(v.si > subC);
    return v.ui | sign;
  }

  CUDAC uint16_t float2half(const volatile float& value) const volatile {  // NOLINT (*)
    Bits v, s;
    v.f = value;
    uint32_t sign = v.si & signN;
    v.si ^= sign;
    sign >>= shiftSign;  // logical shift
    s.si = mulN;
    s.si = s.f * v.f;  // correct subnormals
    v.si ^= (s.si ^ v.si) & -(minN > v.si);
    v.si ^= (infN ^ v.si) & -((infN > v.si) & (v.si > maxN));
    v.si ^= (nanN ^ v.si) & -((nanN > v.si) & (v.si > infN));
    v.ui >>= shift;  // logical shift
    v.si ^= ((v.si - maxD) ^ v.si) & -(v.si > maxC);
    v.si ^= ((v.si - minD) ^ v.si) & -(v.si > subC);
    return v.ui | sign;
  }

  CUDAC float half2float(const uint16_t& value) const {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  CUDAC float half2float(const volatile uint16_t& value) const volatile {  // NOLINT(*)
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  template<typename T>
  CUDAC void constructor(const T& value) {
#if CUDA_HALF
    cuhalf_ = __float2half(float(value));  // NOLINT(*)
#else
    half_ = float2half(float(value));  // NOLINT(*)
#endif  // CUDA_HALF
  }
};

/*! \brief overloaded + operator for half_t */
HALF_OPERATOR(half_t, +)
/*! \brief overloaded - operator for half_t */
HALF_OPERATOR(half_t, -)
/*! \brief overloaded * operator for half_t */
HALF_OPERATOR(half_t, *)
/*! \brief overloaded / operator for half_t */
HALF_OPERATOR(half_t, /)
/*! \brief overloaded > operator for half_t */
HALF_OPERATOR(bool, >)
/*! \brief overloaded < operator for half_t */
HALF_OPERATOR(bool, <)
/*! \brief overloaded >= operator for half_t */
HALF_OPERATOR(bool, >=)
/*! \brief overloaded <= operator for half_t */
HALF_OPERATOR(bool, <=)

}  // namespace half
} // namespace uni

#endif
#endif  // HALF_H_

