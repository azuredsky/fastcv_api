/**
 * @file vector.hpp
 *
 * @brief vector related operations
 *
 * @note Only support X86, CUDA and ARM, OCL is not supported
 */

#ifndef HPC_FMATH_VECTOR_H_
#define HPC_FMATH_VECTOR_H_
#include <stdlib.h>
#include <stdio.h>
#include "sys.h"

#if defined(USE_OCL)
#include "bcl.h"
#include "bclCommon.hpp"
#include "kernelCache.hpp"
#endif

namespace HPC { namespace fmath {
#if defined(USE_X86)
    /**
     * @brief arithmetic operator y += ax
     */
    template<typename T>
        void x86VectorAxpy(size_t len, T a, const T* x, T* y);

    /**
     * @brief sparse vector multiplication y += ax
     * @param numNotZeros: num of non-zeros in vector xv
     * @param xIndex: index of vector xv
     */
    template<typename T>
        void x86SparseVectorAxpy(size_t numNotZeros, T av, const int *xIndex, const T* xv, T* y);

    /**
     * @brief arithmetic operator y = ax + b
     * @param len: the lenght of x and y
     */
    template<typename T>
        void x86VectorAxpb(size_t len, T a, const T* x, T b, T* y);

    /**
     * @brief arithmetic operator y = ax + by
     * @param len: the lenght of x and y
     */
    template<typename T>
        void x86VectorAxpby(size_t len, T a, const T* x, T b, T* y);

    /**
     * @brief vector add out = a + b
     */
    template<typename T>
        void x86VectorAdd(size_t len, const T* a, const T* b, T* out);

    /**
     * @brief vector add scalar by elements
     */
    template<typename T>
        void x86VectorAddScalar(size_t len, const T* a, T b, T* out);

    /**
     * @brief vector copy operation
     */
    template<typename T>
        void x86VectorCopy(size_t len, const T* in, T* out);

    /**
     * @brief vector division by elements
     */
    template<typename T>
        void x86VectorDiv(size_t len, const T* a, const T* b, T* out) ;

    /**
     * @brief square vector by elements T = sum(x.x)
     */
    template<typename T>
        T x86VectorDot(size_t len, const T* x);

    /**
     * @brief get the Exp of vector by elements out(i) = exp(in(i))
     */
    template<typename T>
        void x86VectorExp(size_t len, const T* in, T* out);

    /**
     * @brief fill x with scalar a
     */
    template<typename T>
        void x86VectorFillScalar(size_t len, T a, T* x);

    /** @brief fill x with vector a
     *  @param N: 2 3 4
     */
    template<typename T, int N>
        void x86VectorFillVector(size_t len, const T a[N], T* x);

    /**
     * @brief get the max element of vector T = max(x)
     */
    template<typename T>
        T x86VectorMax(size_t len, const T* x);

    /**
     * @brief get the min element of vector T = min(x)
     */
    template<typename T>
        T x86VectorMin(size_t len, const T* x);

    /**
     * @brief get the max and min element of vector mi = min(x) ma = max(x)
     */
    template<typename T>
        void x86VectorMaxMin(size_t len, const T* x, T* mi, T* ma);

    /**
     * @brief vector dot multiplication and additive operation out += a.b
     * @param len: the lenght of vector a b and out
     */
    template<typename T>
        void x86VectorMla(size_t len, const T* a, const T* b, T* out);

    /**
     * @brief vector dot multiplication and subtraction operation out -= a.b
     * @param len: the lenght of vector a b and out
     */
    template<typename T>
        void x86VectorMls(size_t len, const T* a, const T* b, T* out);

    /**
     * @brief vector dot multiplication out = a.b
     * @param len: the lenght of vector a b and out
     */
    template<typename T>
        void x86VectorMul(size_t len, const T* a, const T* b, T* out);

    /**
     * @brief vector scalar out = ba
     */
    template<typename T>
        void x86VectorMulScalar(size_t len, const T* a, T b, T* out);

    /**
     * @brief vector mask with threshold and scale by elements
     */
    template<typename T>
        void x86VectorMaskAndScale(size_t len, const T* mask, const T alpha, const T* in,
                T threshold, T scale, const T beta, T* out);

    /**
     * @brief get the power of vector by elements
     */
    template<typename T>
        void x86VectorPower(size_t len, const T* in, T beta, T* out);

    //template<typename T>
    //void x86VectorRandGenerateUniform(size_t len, T* mask);

    /**
     * @brief get the relu of vector by elements
     */
    template<typename T>
        void x86VectorRelu(ISA_t isa, IMP_t imp,size_t len, const T alpha, const T* in, const T beta, T* out);

    /**
     * @brief relu backward
     */
    template<typename T>
        void x86VectorReluBackward(size_t len, const T alpha, const T* in, const T* gradOut, const T beta, T* gradIn);

    /**
     * @brief randon generate vector with the elements ranged in [0 1)
     */
#ifdef USE_CPP11
    template<typename T>
        void x86VectorRandGenerate(size_t len, T* mask);
#endif

    /**
     * @brief swap x and y with corresponding scale
     */
    template<typename T>
        void x86VectorSwap(size_t len, T scaleX, T* x, T scaleY, T* y);

    //TODO
    template<typename T>
        void x86VectorScale(size_t len, T a, const T* x, T* y);

    /**
     * @brief get sum of vector T = sum(x)
     */
    template<typename T>
        T x86VectorSum(size_t len, const T* x);

    /**
     * @brief get the sigmoid of vector by elements
     */
    template<typename T>
        void x86VectorSigmoid(size_t len, const T alpha, const T* in, const T beta, T* out);

    /**
     * @brief sigmoid backward
     */
    template<typename T>
        void x86VectorSigmoidBackward(size_t len, const T alpha, const T* out, const T* gradOut, const T beta, T* gradIn);

    /**
     * @brief get the squre of vector by elements
     */
    template<typename T>
        void x86VectorSqure(size_t len, const T* in, T* out);

    /**
     * @brief vector subtraction out = a - b
     */
    template<typename T>
        void x86VectorSubtract(size_t len, const T* a, const T* b, T* out);

    /**
     * @brief vector split backward
     */
    template<typename T>
        void x86VectorSplitBackward(size_t len, const T alpha, const T* a, const T* b,
                const T beta, T* out);

    /**
     * @brief get the tanh of vector by elements
     */
    template<typename T>
        void x86VectorTanh(size_t len, const T alpha, const T* in, const T beta, T* out);

    /**
     * @brief tanh backward
     */
    template<typename T>
        void x86VectorTanhBackward(size_t num, const T alpha, const T* out, const T* gradOut, const T beta, T* gradIn);
#endif //USE_X86
#if defined(USE_ARM)
    /**
     * @brief arithmetic operator y += ax
     */
    template<typename T>
        void armVectorAxpy(size_t len, T a, const T* x, T* y);

    /**
     * @brief sparse vector multiplication y += ax
     * @param numNotZeros: num of non-zeros in vector xv
     * @param xIndex: index of vector xv
     */
    template<typename T>
        void armSparseVectorAxpy(size_t numNotZeros, T av, const int *xIndex, const T* xv, T* y);

    /**
     * @brief arithmetic operator y = ax + b
     * @param len: the lenght of x and y
     */
    template<typename T>
        void armVectorAxpb(size_t len, T a, const T* x, T b, T* y);

    /**
     * @brief arithmetic operator y = ax + by
     * @param len: the lenght of x and y
     */
    template<typename T>
        void armVectorAxpby(size_t len, T a, const T* x, T b, T* y);

    /**
     * @brief vector add out = a + b
     */
    template<typename T>
        void armVectorAdd(size_t len, const T* a, const T* b, T* out);

    /**
     * @brief vector add scalar by elements out = a + Extend(b)
     */
    template<typename T>
        void armVectorAddScalar(size_t len, const T* a, T b, T* out);

    /**
     * @brief vector copy operation
     */
    template<typename T>
        void armVectorCopy(size_t len, const T* in, T* out);

    /**
     * @brief square vector by elements T = sum(x.x)
     */
    template<typename T>
        T armVectorDot(size_t len, const T* x);

    /**
     * @{
     */
    /**
     * @brief vector division by elements
     */
    template<typename T>
        void armVectorDiv(size_t len, const T* a, const T* b, T* out);

    template<typename T>
        void armVectorDivScalar(size_t len, const T* a, const T* b, T* out);
    /**
     * @}
     */

    /**
     * @brief get the Exp of vector by elements out(i) = exp(in(i))
     */
    template<typename T>
        void armVectorExp(size_t len, const T* in, T* out);

    /**
     * @brief fill x with scalar a
     */
    template<typename T>
        void armVectorFillScalar(size_t len, T a, T* x);

    /**
     * @brief fill x with vector a
     * @param N: 2 3 4
     */
    template<typename T, int N>
        void armVectorFillVector(size_t len, const T a[N], T* x);

    /**
     * @brief get the max element of vector T = max(x)
     */
    template<typename T>
        T armVectorMax(size_t len, const T* x);

    /**
     * @brief get the min element of vector T = min(x)
     */
    template<typename T>
        T armVectorMin(size_t len, const T* x);

    /**
     * @brief get the max and min element of vector mi = min(x) ma = max(x)
     */
    template<typename T>
        void armVectorMaxMin(size_t len, const T* x, T* mi, T* ma);

    /**
     * @brief vector dot multiplication and additive operation out += a.b
     * @param len: the lenght of vector a b and out
     */
    template<typename T>
        void armVectorMla(size_t len, const T* a, const T* b, T* out);

    /**
     * @brief vector dot multiplication and subtraction operation out -= a.b
     * @param len: the lenght of vector a b and out
     */
    template<typename T>
        void armVectorMls(size_t len, const T* a, const T* b, T* out);

    /**
     * @brief vector dot multiplication out = a.b
     * @param len: the lenght of vector a b and out
     */
    template<typename T>
        void armVectorMul(size_t len, const T* a, const T* b, T* out);

    /**
     * @brief vector scalar out = ba
     */
    template<typename T>
        void armVectorMulScalar(size_t len, const T* a, const T b, T* out);

    /**
     * @brief vector mask with threshold and scale by elements
     */
    template<typename T>
        void armVectorMaskAndScale(size_t len, const T* mask, const T alpha, const T* in,
                T threshold, T scale, const T beta, T* out);

    /**
     * @brief get the power of vector by elements
     */
    template<typename T>
        void armVectorPower(size_t len, const T* in, T beta, T* out);

    //template<typename T>
    //void armVectorRandGenerateUniform(size_t len, T* mask);

    /**
     * @brief get the relu of vector by elements
     */
    template<typename T>
        void armVectorRelu(size_t len, const T alpha, const T* in, const T beta, T* out);

    /**
     * @brief relu backward
     */
    template<typename T>
        void armVectorReluBackward(size_t len, const T alpha, const T* in, const T* gradOut,
                const T beta, T* gradIn);

    /**
     * @brief randon generate vector with the elements ranged in [0 1)
     */
    template<typename T>
        void armVectorRandGenerate(size_t len, T* mask);

    /**
     * @brief swap x and y with corresponding scale
     */
    template<typename T>
        void armVectorSwap(size_t len, T scaleX, T* x, T scaleY, T* y);

    /**
     * @brief get the squre of vector by elements
     */
    template<typename T>
        void armVectorSqure(size_t len, const T* in, T* out);

    /**
     * @brief vector subtraction out = a - b
     */
    template<typename T>
        void armVectorSubtract(size_t len, const T* a, const T* b, T* out);

    /**
     * @{
     */
    /**
     * @brief vector split backward
     */
    template<typename T>
        void armVectorSplitBackward(size_t len, const T alpha, const T* a, const T* b,
                const T beta, T* out);
    //TODO
    template<typename T>
        void armVectorScale(size_t len, T a, const T* x, T* y);
    /**
     * @}
     */

    /**
     * @brief get sum of vector T = sum(x)
     */
    template<typename T>
        T armVectorSum(size_t len, const T* x);

    /**
     * @brief get the sigmoid of vector by elements
     */
    template<typename T>
        void armVectorSigmoid(size_t len, const T alpha, const T* in, const T beta, T* out);

    /**
     * @brief sigmoid backward
     */
    template<typename T>
        void armVectorSigmoidBackward(size_t len, const T alpha, const T* out, const T* gradOut,
                const T beta, T* gradIn);

    /**
     * @brief get the tanh of vector by elements
     */
    template<typename T>
        void armVectorTanh(size_t len, const T alpha, const T* in, const T beta, T* out);

    /**
     * @brief tanh backward
     */
    template<typename T>
        void armVectorTanhBackward(size_t len, const T alpha, const T* out, const T* gradOut,
                const T beta, T* gradIn);

#endif //USE_ARM
#if defined(USE_CUDA)
    /**
     * @brief arithmetic operator y += ax
     */
    template<typename T>
        void cudaVectorAxpy(int device, cudaStream_t stream, size_t len, T a, const T* x, T* y);

    /**
     * @brief sparse vector multiplication y += ax
     * @param numNotZeros: num of non-zeros in vector xv
     * @param xIndex: index of vector xv
     */
    template<typename T>
        void cudaSparseVectorAxpy(int device, cudaStream_t stream,
                size_t numNotZeros, T av, const int *xIndex, const T* xv, T* y);

    /**
     * @brief arithmetic operator y = ax + b
     * @param len: the lenght of x and y
     */
    template<typename T>
        void cudaVectorAxpb(int device, cudaStream_t stream, size_t len, T a, const T* x, T b, T* y);

    /**
     * @brief arithmetic operator y = ax + by
     * @param len: the lenght of x and y
     */
    template<typename T>
        void cudaVectorAxpby(int device, cudaStream_t stream, size_t len, T a, const T* x, T b, T* y);

    /**
     * @brief vector add out = a + b
     */
    template<typename T>
        void cudaVectorAdd(int device, cudaStream_t stream, size_t len, const T *a, const T *b, T *c);

    /**
     * @brief vector add scalar by elements
     */
    template<typename T>
        void cudaVectorAddScalar(int device, cudaStream_t stream, size_t len, const T *a, T b, T *c);

    /**
     * @brief comput diff with scale
     */
    template<typename T>
        void cudaVectorComputDiffAndScale(int device, cudaStream_t stream,
                size_t len, T scale, const T* d1, const T* d2, T* diff);

    /**
     * brief copy
     */
    template<typename T>
        void cudaVectorCopy(int device, cudaStream_t stream, size_t len, const T* in, T* out);

    /**
     * @brief square vector by elements T = sum(x.x)
     */
    template<typename T>
        T cudaVectorDot(int device, cudaStream_t stream, size_t len, const T* x);

    /**
     * @brief vector division by elements
     */
    template<typename T>
        void cudaVectorDiv(int device, cudaStream_t stream, size_t len, const T *a, const T *b, T *c) ;

    /**
     * @brief get the Exp of vector by elements out(i) = exp(in(i))
     */
    template <typename T>
        void cudaVectorExp(int device, cudaStream_t stream, size_t len, const T* in, T* out);

    /**
     * @brief fill x with scalar a
     */
    template<typename T>
        void cudaVectorFillScalar(int device, cudaStream_t stream, size_t len, T a, T* x);

    /**
     * @brief fill x with vector a
     * @param N: 2 3 4
     */
    template<typename T, int N>
        void cudaVectorFillVector(int device, cudaStream_t stream, size_t len, const T a[N], T* x);

    /**
     * @brief vector fill constant with value v
     */
    template<typename T>
        void cudaVectorFillConstant(int device, cudaStream_t stream,  size_t len, T v, T* ptr);

    /**
     * @brief get the max element of vector T = max(x)
     */
    template<typename T>
        T cudaVectorMax(int device, cudaStream_t stream, size_t len, const T* x);

    /**
     * @brief get the min element of vector T = min(x)
     */
    template<typename T>
        T cudaVectorMin(int device, cudaStream_t stream, size_t len, const T* x);

    /**
     * @brief get the max and min element of vector mi = min(x) ma = max(x)
     */
    template<typename T>
        void cudaVectorMaxMin(int device, cudaStream_t stream, size_t len, const T* x, T* mi, T* ma);

    /**
     * @brief vector mask with threshold and scale by elements
     */
    template<typename T>
        void cudaVectorMaskAndScale(int device, cudaStream_t stream, size_t len,
                const T alpha, const T* input, const T* mask, float ratio, float scale, const T beta, T* output);

    /**
     * @brief vector dot multiplication and additive operation out += a.b
     * @param len: the lenght of vector a b and out
     */
    template<typename T>
        void cudaVectorMla(int device, cudaStream_t stream, size_t len, const T *a, const T *b, T *c) ;

    /**
     * @brief vector dot multiplication and subtraction operation out -= a.b
     * @param len: the lenght of vector a b and out
     */
    template<typename T>
        void cudaVectorMls(int device, cudaStream_t stream, size_t len, const T *a, const T *b, T *c) ;

    /**
     * @brief vector dot multiplication out = a.b
     * @param len: the lenght of vector a b and out
     */
    template<typename T>
        void cudaVectorMul(int device, cudaStream_t stream, size_t len, const T *a, const T *b, T *c) ;

    /**
     * @brief vector scalar out = ba
     */
    template<typename T>
        void cudaVectorMulScalar(int device, cudaStream_t stream, size_t len, const T *a, T b, T *c) ;

    /**
     * @brief get the power of vector by elements
     */
    template<typename T>
        void cudaVectorPower(int device, cudaStream_t stream, size_t len, const T *x, T alpha, T *y);

    /**
     * @brief swap x and y with corresponding scale
     */
    template<typename T>
        void cudaVectorSwap(int device, cudaStream_t stream, size_t len, T scaleX, T* x, T scaleY, T* y);

    /**
     * @brief vector scale
     */
    template<typename T>
        void cudaVectorScale(int device, cudaStream_t stream, size_t len, T a, const T* x, T* y);

    /**
     * @brief get sum of vector T = sum(x)
     */
    template<typename T>
        T cudaVectorSum(int device, cudaStream_t stream, size_t len, const T* x);

    /**
     * @brief vector subtraction out = a - b
     */
    template<typename T>
        void cudaVectorSubtract(int device, cudaStream_t stream, size_t len, const T *a, const T *b, T *c);

    /**
     * @brief vector split backward
     */
    template<typename T>
        void cudaVectorSplitBackward(int device, cudaStream_t stream, size_t len,
                const T alpha, const T *a, const T *b, const T beta, T *c);
#endif //USE_CUDA
#if defined(USE_OCL)

    /**
     * @brief sparse vector multiplication y += ax
     * @param numNotZeros: num of non-zeros in vector xv
     * @param xIndex: index of vector xv
     */
    template<typename T>
        void oclSparseVectorAxpy(cl_command_queue queue, cl_uint numNotZeros, T av, const cl_mem xIndex, const cl_mem xv, cl_mem y);

    /**
     * @brief arithmetic operator y += ax
     */
    template<typename T>
        void oclVectorAxpy(cl_command_queue queue, cl_uint len, T a, const cl_mem x, cl_mem y);


    /* @Brief arithmetic operator y = ax + b
     * @Param len: the lenght of x and y
     */
    template<typename T>
        void oclVectorAxpb(cl_command_queue queue, cl_uint len, T a, const cl_mem x, T b, cl_mem y);

    /* @Brief arithmetic operator y = ax + by
     * @Param len: the lenght of x and y
     */
    template<typename T>
        void oclVectorAxpby(cl_command_queue queue, cl_uint len, T a, const cl_mem x, T b, cl_mem y);

    /* @Brief arithmetic operator y = x 
     * @Param len: the lenght of x and y
     */
    template<typename T> 
        void oclVectorCopy(cl_command_queue queue, cl_uint len, const cl_mem x, cl_mem y);

    /* @Brief square vector by elements T = sum(x.x) */
    template<typename T> 
        T oclVectorDot(cl_command_queue queue, cl_uint len, const cl_mem x); 

    /* @Brief fill x with scalar a */
    template<typename T>
        void oclVectorFillScalar(cl_command_queue queue, cl_uint len, T a, cl_mem x);

    /* @Brief fill x with vector a
     * @Param N: 2 3 4
     */
    template<typename T, int N>
        void oclVectorFillVector(cl_uint len, const T a[N], cl_mem x);

    /* @Brief get the max element of vector T = max(x) */
    template<typename T> 
        T oclVectorMax(cl_command_queue queue, cl_uint len, const cl_mem x); 

    /* @Brief get the min element of vector T = min(x) */
    template<typename T>
        T oclVectorMin(cl_command_queue queue, cl_uint len, const cl_mem x);

    /**
     * @brief get the max and min element of vector mi = min(x) ma = max(x)
     */
    template<typename T>
        void oclVectorMaxMin(cl_command_queue queue, cl_uint len, const T* x, T* mi, T* ma);

    /* @Brief swap x and y with corresponding scale */
    template<typename T> 
        void oclVectorSwap(cl_command_queue queue, cl_uint len, T scaleX, const cl_mem x, T scaleY, cl_mem y);

    //TODO
    template<typename T> 
        void oclVectorScal(cl_command_queue queue, cl_uint len, T a, cl_mem x);

    /* @Brief get sum of vector T = sum(x) */
    template<typename T> 
        T oclVectorSum(cl_command_queue queue, cl_uint len, const cl_mem x);

    /* @Brief get the tanh result of vector */
    template<typename T> 
        void oclVectorSigmoid(cl_command_queue queue,  cl_uint len, T alpha, const cl_mem in, T beta, cl_mem out);

    /* @Brief get the tanh backward result of vector */
    template<typename T> 
        void oclVectorSigmoidBackward(cl_command_queue queue, cl_uint len, T alpha, const cl_mem out, const cl_mem gradOut, T beta, cl_mem gradIn);

    /* @Brief get the tanh result of vector */
    template<typename T> 
        void oclVectorRelu(cl_command_queue queue,  cl_uint len, T alpha, const cl_mem in, T beta, cl_mem out);

    /* @Brief get the tanh backward result of vector */
    template<typename T> 
        void oclVectorReluBackward(cl_command_queue queue, cl_uint len, T alpha, const cl_mem out, const cl_mem gradOut, T beta, cl_mem gradIn);

    /* @Brief get the tanh result of vector */
    template<typename T> 
        void oclVectorTanh(cl_command_queue queue,  cl_uint len, T alpha, const cl_mem in, T beta, cl_mem out);

    /* @Brief get the tanh backward result of vector */
    template<typename T> 
        void oclVectorTanhBackward(cl_command_queue queue, cl_uint len, T alpha, const cl_mem out, const cl_mem gradOut, T beta, cl_mem gradIn);

#endif //USE_OCL
} }; //end namespace HPC fmath

#endif
