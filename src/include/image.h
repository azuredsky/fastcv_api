#ifndef HPC_IMAGE_H_
#define HPC_IMAGE_H_
#include <string>
#include <cstring>
#include <cmath>
#include <cassert>

#if defined(USE_CUDA)
#include <cublas_v2.h>
#include <half.h>
using namespace uni::half;
#endif

#if defined(USE_ARM)
#include <arm_neon.h>
#endif

#if defined(USE_OCL)
#include <CL/opencl.h>
#include "bcl.h"
#include "kernelCache.hpp"
#endif



namespace HPC { namespace fastcv {
    //@{
    /**
     * @brief read and write
     *
     *
     * available now(T):
     * arm: (unsigned char) (int) (unsigned int) (float)
     **/
/*
#if defined(USE_X86)
#endif
#if defined(USE_ARM)
    template<int nc, typename T> struct DT;
    template<> struct DT<1, unsigned char> {
        typedef uint8x8_t vec_DT;
    };
    template<> struct DT<3, unsigned char> {
        typedef uint8x8x3_t vec_DT;
    };
    template<> struct DT<4, unsigned char> {
        typedef uint8x8x4_t vec_DT;
    };
    template<> struct DT<1, float> {
        typedef float32x4_t vec_DT;
    };
    template<> struct DT<3, float> {
        typedef float32x4x3_t vec_DT;
    };
    template<> struct DT<4, float> {
        typedef float32x4x4_t vec_DT;
    };


    template<int nc, typename T> struct DT16;
    template<> struct DT16<3, unsigned char> {
        typedef uint8x16x3_t vec_DT;
    };
    template<> struct DT16<4, unsigned char> {
        typedef uint8x16x4_t vec_DT;
    };

    template<int nc, typename Tptr, typename T> inline void vstx_u8_f32(Tptr *ptr, T vec);
    template<int nc, typename Tptr, typename T> inline T vldx_u8_f32(const Tptr *ptr);

    template<int nc, typename T> inline void vzero_u8_f32(T* ptr);
    template<int nc, typename T, typename T_val> inline void vmla_u8_f32(T* dst, T src0, T_val src1);
    template<int nc, typename T> inline void vadd_u8_f32(T* dst, T src);

    template<int nc, typename T> inline void mla_ptr_pixel(T * dst, const T * src0, T src1);
    template<int nc, typename T> inline void stx_pixel(T *ptr, T *val);
#endif
#if defined(USE_CUDA)
#endif
#if defined(USE_OCL)
#endif
    //@}
*/
    //@{
    /**
     * @brief merge SOA(struct 0f array) to AOS(array of struct)
     *
     * @param numChannels           input Data's channels
     * @param height                input Data's height
     * @param width                 input Data's width need to be
     * processed
     * @param inWidthStride         input Data's width
     * @param inData                input Data
     * @param outWidthStride        output Data's width
     * @param outData               output Data
     *
     * available now(T):
     * x86: (unsigned char) (int) (unsigned int) (float)
     * state: in visual studio on Windows, template does not support the template
               specialization with genner array, such that:

               template<typename T, int len>
                    void fun(T a[len])
                    {
                        for (int i = 0; i < len; i++);
                    }
               template<>
                    void fun<float, 3>(float a[3])
                    {
                        for (int i = 0; i < 3; i++);
                    }

              it will get the error error C2912: Explicit specialty；“void fun<float,3>(float [])” is
               not the specialty of function template.
              thus here use pointer's pointer T **inData replace pointer array for T *inData[nc].
              it's also the same to function SplitNChannelsImage.

     * arm: (unsigned char) (int) (unsigned int) (float)
     * cuda: (unsigned char) (int) (unsigned int) (float)
     **/
#if defined(USE_X86)
    template<typename T, int nc>
        void x86MergeNChannelsImage(int height, int width,
                int inWidthStride, const T** inData,
                int outWidthStride, T* outData);
#endif
#if defined(USE_ARM)
    template<typename T, int nc>
        void armMergeNChannelsImage(int height, int width,
                int inWidthStride, const T** inData,
                int outWidthStride, T* outData);
#endif
#if defined(USE_CUDA)
    template<typename T, int nc>
        void cudaMergeNChannelsImage(int device, cudaStream_t stream,
                int height, int width,
                int inWidthStride, const T** inData,
                int outWidthStride, T* outData);
#endif
#if defined(USE_OCL)
    template<typename T, int nc>
        void oclMergeNChannelsImage(cl_command_queue queue,
                int height, int width,
                int inWidthStride, const cl_mem inData[nc],
                int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
    /**
     * @brief split AOS(array of struct) to SOA(struct of array)
     * it looks like a matrix transpose
     *
     * @param height                input Data's height
     * @param width                 input Data's width need to be
     * processed
     * @param inWidthStride         input Data's width
     * @param numChannels           input Data's channels
     * @param inData                input Data
     * @param outWidthStride        output Data's width
     * @param outData               output Data
     *
     * available now(T):
     * x86: (unsigned char) (int) (unsigned int) (float)
     * state: the case same with commit in MergeNChannelsImage

     * arm: (unsigned char) (int) (unsigned int) (float)
     * cuda: (unsigned char) (int) (unsigned int) (float)
     **/
#if defined(USE_X86)
    template<typename T, int nc>
        void x86SplitNChannelsImage(int height, int width,
                int inWidthStride, const T* inData,
                int outWidthStride, T** outData);
#endif
#if defined(USE_ARM)
    template<typename T, int nc>
        void armSplitNChannelsImage(int height, int width,
                int inWidthStride, const T* inData,
                int outWidthStride, T** outData);
#endif
#if defined(USE_CUDA)
    template<typename T, int nc>
        void cudaSplitNChannelsImage(int device, cudaStream_t stream,
                int height, int width,
                int inWidthStride, const T* inData,
                int outWidthStride, T** outData);
#endif
#if defined(USE_OCL)
    template<typename T, int nc>
        void oclSplitNChannelsImage(cl_command_queue queue,
                int height, int width,
                int inWidthStride, const cl_mem inData,
                int outWidthStride, cl_mem outData[nc]);
#endif
    //@}

    //@{
    /**
     * @brief convert BGR format image to GRAY format
     * number elements of inDesc and outDesc should be the same
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outWidthStride    input Data's width
     * @param outData           output Data
     *
     * available now (type channels):
     * x86: ()
     * arm: () ()
     * cuda: (uchar 4)
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void x86BGR2GRAYImage(
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void armBGR2GRAYImage(int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void cudaBGR2GRAYImage(int device, cudaStream_t stream,
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void oclBGR2GRAYImage(cl_command_queue command_queue,
                int height, int width,
                int inWidthStride, const cl_mem inData,
                int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
    /**
     * @brief convert GRAY format image to BGR format
     * number elements of inDesc and outDesc should be the same
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outWidthStride    input Data's width
     * @param outData           output Data
     *
     * available now (type channels):
     * x86: ()
     * arm: () ()
     * cuda: (uchar 4)
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void x86GRAY2BGRImage(
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void armGRAY2BGRImage(int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void cudaGRAY2BGRImage(int device, cudaStream_t stream,
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void oclGRAY2BGRImage(cl_command_queue command_queue,
                int height, int width,
                int inWidthStride, const cl_mem inData,
                int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
    /**
     * @brief convert GBR format image to GBRA format
     * number elements of inDesc and outDesc should be the same
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outWidthStride    input Data's width
     * @param outData           output Data
     *
     * available now (type channels):
     * x86: ()
     * arm: () ()
     * cuda: (uchar 4)
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void x86BGR2BGRAImage(
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void armBGR2BGRAImage(int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void cudaBGR2BGRAImage(int device, cudaStream_t stream,
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void oclBGR2BGRAImage(cl_command_queue command_queue,
                int height, int width,
                int inWidthStride, const cl_mem inData,
                int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
    /**
     * @brief convert BGRA format image to BGR format
     * number elements of inDesc and outDesc should be the same
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outWidthStride    input Data's width
     * @param outData           output Data
     *
     * available now (type channels):
     * x86: ()
     * arm: () ()
     * cuda: (uchar 4)
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void x86BGRA2BGRImage(
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void armBGRA2BGRImage(int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void cudaBGRA2BGRImage(int device, cudaStream_t stream,
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void oclBGRA2BGRImage(cl_command_queue command_queue,
                int height, int width,
                int inWidthStride, const cl_mem inData,
                int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
    /**
     * @brief convert BGR(A) format image to YCrCb format
     * number elements of inDesc and outDesc should be the same
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outWidthStride    input Data's width
     * @param outData           output Data
     * available now (type channels):
     * x86: (uchar 3,4) (float 3,4)
     * arm: (uchar 3,4) (float 3,4)
     * cuda: (uchar 3,4) (float 3,4)
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void x86BGR2YCrCbImage(int height, int width, int inWidthStride, const Tsrc* inData, int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void armBGR2YCrCbImage(int height, int width, int inWidthStride, const Tsrc* inData, int outWidthStride, Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void cudaBGR2YCrCbImage(int device, cudaStream_t stream, int height, int width, int inWidthStride, const Tsrc* inData, int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void oclBGR2YCrCbImage(cl_command_queue command_queue,int height, int width, int inWidthStride, const cl_mem inData, int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
    /**
     * @brief convert YCrCb format image to BGR(A) format
     * number elements of inDesc and outDesc should be the same
     *
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outWidthStride    input Data's width
     * @param outData           output Data
     *
     * available now (type channels):
     * x86: (uchar 3,4) (float 3,4)
     * arm: (uchar 3,4) (float 3,4)
     * cuda: (uchar 3,4) (float 3,4)
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void x86YCrCb2BGRImage(int height, int width, int inWidthStride, const Tsrc* inData, int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void armYCrCb2BGRImage(int height, int width, int inWidthStride, const Tsrc* inData, int outWidthStride, Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void cudaYCrCb2BGRImage(int device, cudaStream_t stream, int height, int width, int inWidthStride, const Tsrc* inData, int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void oclYCrCb2BGRImage(cl_command_queue command_queue,int height, int width, int inWidthStride, const cl_mem inData, int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
    /**
     * @brief convert BGR(A) format image to HSV format
     * number elements of inDesc and outDesc should be the same
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outWidthStride    input Data's width
     * @param outData           output Data
     * available now (type channels):
     * x86: (uchar 3,4) (float 3,4)
     * arm: (uchar 3,4) (float 3,4)
     * cuda: (uchar 3,4) (float 3,4)
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void x86BGR2HSVImage(int height, int width, int inWidthStride, const Tsrc* inData, int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void armBGR2HSVImage(int height, int width, int inWidthStride, const Tsrc* inData, int outWidthStride, Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void cudaBGR2HSVImage(int device, cudaStream_t stream, int height, int width, int inWidthStride, const Tsrc* inData, int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void oclBGR2HSVImage(cl_command_queue command_queue,int height, int width, int inWidthStride, const cl_mem inData, int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
    /**
     * @brief convert HSV format image to BGR(A) format
     * number elements of inDesc and outDesc should be the same
     *
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outWidthStride    input Data's width
     * @param outData           output Data
     *
     * available now (type channels):
     * x86: (uchar 3,4) (float 3,4)
     * arm: (uchar 3,4) (float 3,4)
     * cuda: (uchar 3,4) (float 3,4)
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void x86HSV2BGRImage(int height, int width, int inWidthStride, const Tsrc* inData, int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void armHSV2BGRImage(int height, int width, int inWidthStride, const Tsrc* inData, int outWidthStride, Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void cudaHSV2BGRImage(int device, cudaStream_t stream, int height, int width, int inWidthStride, const Tsrc* inData, int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void oclHSV2BGRImage(cl_command_queue command_queue,int height, int width, int inWidthStride, const cl_mem inData, int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
    /**
     * @brief convert BGR(A) format image to LAB format
     * number elements of inDesc and outDesc should be the same
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outWidthStride    input Data's width
     * @param outData           output Data
     * available now (type channels):
     * x86: (uchar 3,4) (float 3,4)
     * arm: (uchar 3,4) (float 3,4)
     * cuda: (uchar 3,4) (float 3,4)
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void x86BGR2LABImage(int height, int width, int inWidthStride, const Tsrc* inData, int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void armBGR2LABImage(int height, int width, int inWidthStride, const Tsrc* inData, int outWidthStride, Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void cudaBGR2LABImage(int device, cudaStream_t stream, int height, int width, int inWidthStride, const Tsrc* inData, int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void oclBGR2LABImage(cl_command_queue command_queue,int height, int width, int inWidthStride, const cl_mem inData, int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
    /**
     * @brief convert LAB format image to BGR(A) format
     * number elements of inDesc and outDesc should be the same
     *
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outWidthStride    input Data's width
     * @param outData           output Data
     *
     * available now (type channels):
     * x86: (uchar 3,4) (float 3,4)
     * arm: (uchar 3,4) (float 3,4)
     * cuda: (uchar 3,4) (float 3,4)
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void x86LAB2BGRImage(int height, int width, int inWidthStride, const Tsrc* inData, int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void armLAB2BGRImage(int height, int width, int inWidthStride, const Tsrc* inData, int outWidthStride, Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void cudaLAB2BGRImage(int device, cudaStream_t stream, int height, int width, int inWidthStride, const Tsrc* inData, int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void oclLAB2BGRImage(cl_command_queue command_queue,int height, int width, int inWidthStride, const cl_mem inData, int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
    /**
     * @brief convert NV21 format data to BGRA format
     * number elements of inDesc and outDesc should be the same
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outWidthStride    input Data's width
     * @param outData           output Data
     *
     * available now (type channels):
     * x86: ()
     * arm: () ()
     * cuda: (uchar 4)
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void x86NV212BGRImage(
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void armNV212BGRImage(int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void cudaNV212BGRImage(int device, cudaStream_t stream,
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void oclNV212BGRImage(cl_command_queue command_queue,
                int height, int width,
                int inWidthStride, const cl_mem inData,
                int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
    /**
     * @brief convert BGR(A) format image to NV21 format
     * number elements of inDesc and outDesc should be the same
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outWidthStride    input Data's width
     * @param outData           output Data
     *
     * available now (type channels):
     * x86: ()
     * arm: () ()
     * cuda: (uchar 4)
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void x86BGR2NV21Image(
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void armBGR2NV21Image(int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void cudaBGR2NV21Image(int device, cudaStream_t stream,
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void oclBGR2NV21Image(cl_command_queue command_queue,
                int height, int width,
                int inWidthStride, const cl_mem inData,
                int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
    /**
     * @brief convert GBR(A) format image to RGB(A) format
     * number elements of inDesc and outDesc should be the same
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outWidthStride    input Data's width
     * @param outData           output Data
     *
     * available now (type channels):
     * x86: ()
     * arm: () ()
     * cuda: (uchar 4)
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void x86BGR2RGBImage(
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void armBGR2RGBImage(int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void cudaBGR2RGBImage(int device, cudaStream_t stream,
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void oclBGR2RGBImage(cl_command_queue command_queue,
                int height, int width,
                int inWidthStride, const cl_mem inData,
                int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
    /**
     * @brief convert NV12 format data to BGRA format
     * number elements of inDesc and outDesc should be the same
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outWidthStride    input Data's width
     * @param outData           output Data
     *
     * available now (type channels):
     * x86: ()
     * arm: () ()
     * cuda: (uchar 4)
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void x86NV122BGRImage(
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void armNV122BGRImage(int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void cudaNV122BGRImage(int device, cudaStream_t stream,
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void oclNV122BGRImage(cl_command_queue commandQueue,
                int height, int width,
                int inWidthStride, const cl_mem inData,
                int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
    /**
     * @brief convert BGR(A) format image to NV12 format
     * number elements of inDesc and outDesc should be the same
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outWidthStride    input Data's width
     * @param outData           output Data
     *
     * available now (type channels):
     * x86: ()
     * arm: () ()
     * cuda: (uchar 4)
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void x86BGR2NV12Image(
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void armBGR2NV12Image(int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void cudaBGR2NV12Image(int device, cudaStream_t stream,
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void oclBGR2NV12Image(cl_command_queue commandQueue,
                int height, int width,
                int inWidthStride, const cl_mem inData,
                int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
    /**
     * @brief convert BGR(A) format image to I420 format
     * number elements of inDesc and outDesc should be the same
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outWidthStride    input Data's width
     * @param outData           output Data
     *
     * available now (type channels):
     * x86: ()
     * arm: () ()
     * cuda: (uchar 4)
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void x86BGR2I420Image(int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void armBGR2I420Image(int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void cudaBGR2I420Image(int device, cudaStream_t stream,
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void oclBGR2I420Image(cl_command_queue commandQueue,
                int height, int width,
                int inWidthStride, const cl_mem inData,
                int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
    /**
     * @brief convert I420 format data to BGRA format
     * number elements of inDesc and outDesc should be the same
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outWidthStride    input Data's width
     * @param outData           output Data
     *
     * available now (type channels):
     * x86: ()
     * arm: () ()
     * cuda: (uchar 4)
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void x86I4202BGRImage(int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void armI4202BGRImage(int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void cudaI4202BGRImage(int device, cudaStream_t stream,
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void oclI4202BGRImage(cl_command_queue commandQueue,
                int height, int width,
                int inWidthStride, const cl_mem inData,
                int outWidthStride, cl_mem outData);
#endif

    //@{
    /**
     * @brief convert NV21 format data to RGB(A) format
     * number elements of inDesc and outDesc should be the same
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outWidthStride    input Data's width
     * @param outData           output Data
     *
     * available now (type channels):
     * x86: ()
     * arm: () ()
     * cuda: (uchar 4)
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void x86NV212RGBImage(
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void armNV212RGBImage(int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void cudaNV212RGBImage(int device, cudaStream_t stream,
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void oclNV212RGBImage(cl_command_queue commandQueue,
                int height, int width,
                int inWidthStride, const cl_mem inData,
                int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
    /**
     * @brief convert NV12 format data to RGB(A) format
     * number elements of inDesc and outDesc should be the same
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outWidthStride    input Data's width
     * @param outData           output Data
     *
     * available now (type channels):
     * x86: ()
     * arm: () ()
     * cuda: (uchar 4)
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void x86NV122RGBImage(
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void armNV122RGBImage(int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void cudaNV122RGBImage(int device, cudaStream_t stream,
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void oclNV122RGBImage(cl_command_queue commandQueue,
                int height, int width,
                int inWidthStride, const cl_mem inData,
                int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
    /**
     * @brief convert RGB(A) format image to NV21 format
     * number elements of inDesc and outDesc should be the same
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outWidthStride    input Data's width
     * @param outData           output Data
     *
     * available now (type channels):
     * x86: ()
     * arm: () ()
     * cuda: (uchar 4)
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void x86RGB2NV21Image(
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void armRGB2NV21Image(int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void cudaRGB2NV21Image(int device, cudaStream_t stream,
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void oclRGB2NV21Image(cl_command_queue commandQueue,
                int height, int width,
                int inWidthStride, const cl_mem inData,
                int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
    /**
     * @brief convert RGB(A) format image to NV12 format
     * number elements of inDesc and outDesc should be the same
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outWidthStride    input Data's width
     * @param outData           output Data
     *
     * available now (type channels):
     * x86: ()
     * arm: () ()
     * cuda: (uchar 4)
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void x86RGB2NV12Image(
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void armRGB2NV12Image(int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void cudaRGB2NV12Image(int device, cudaStream_t stream,
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void oclRGB2NV12Image(cl_command_queue commandQueue,
                int height, int width,
                int inWidthStride, const cl_mem inData,
                int outWidthStride, cl_mem outData);
#endif
    //@}


    //@{
    /**
     * @brief convert YUV format image to GRAY format
     * number elements of inDesc and outDesc should be the same
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outWidthStride    input Data's width
     * @param outData           output Data
     *
     * available now (type channels):
     * x86: ()
     * arm: () ()
     * cuda: (uchar 4)
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void x86YUV2GRAYImage(
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void armYUV2GRAYImage(int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void cudaYUV2GRAYImage(int device, cudaStream_t stream,
                int height, int width,
                int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst>
        void oclYUV2GRAYImage(cl_command_queue commandQueue,
                int height, int width,
                int inWidthStride, const cl_mem inData,
                int outWidthStride, cl_mem outData);
#endif
    //@}



    //@{
    /**
     * @change the type and scale of data
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outWidthStride    output Data's width
     * @param outData           output Data
     *
     * available now (type channels): unsigned char, unsigned
     * short, unsigned int, float, double can convert to each other.
     **/
#if defined(USE_X86)
    template<typename T_src, int ncSrc, typename T_dst, int ncDst, int nc>
        void x86ChangeDataTypeAndScale(int height, int width, int inWidthStride,
                const T_src* inData, float scale, int outWidthStride, T_dst* outData);
#endif
#if defined(USE_ARM)
    template<typename T_src, int ncSrc, typename T_dst, int ncDst, int nc>
        void armChangeDataTypeAndScale(int height, int width, int inWidthStride,
                const T_src* inData, float scale, int outWidthStride, T_dst* outData);
#endif
#if defined(USE_CUDA)
    template<typename T_src, int ncSrc, typename T_dst, int ncDst, int nc>
        void cudaChangeDataTypeAndScale(int device, cudaStream_t stream,
                int height, int width, int inWidthStride,
                const T_src* inData, float scale, int outWidthStride, T_dst* outData);
#endif
#if defined(USE_OCL)
    template<typename T_src, int ncSrc, typename T_dst, int ncDst, int nc>
        void oclChangeDataTypeAndScale(cl_command_queue queue,
                 int height, int width, int inWidthStride,const cl_mem inData,
				 float scale, int outWidthStride, cl_mem outData);
#endif
    //@}

#if defined(USE_X86)
    void x86BilateralFilter();
#endif
#if defined(USE_ARM)
#endif
#if defined(USE_CUDA)
#endif

    //@{
    /**
     * @resize image or data into the dest size of outDesc
     * watch out the element type and the interpolation method of different
     * architecture
     *
     * @param inHeight          input Data's height
     * @param inWidth           input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outHeight         output Data's height
     * @param outWidth          output Data's width need to be processed
     * @param outWidthStride    output Data's width
     * @param outData           output Data
     * @param mode              linear interpolation will be used if mode is INTERPOLATION_TYPE_LINEAR, nearest interpolation will be used if mode is INTERPOLATION_TYPE_NEAREST_POINT
     *
     * available now (type channels mode)
     * x86:(uchar 1 linear) (uchar 4 linear)
     * arm:(ucahr {1,3,4} linear) (float {1,3,4} linear)
     * cuda:({uchar short-int unsigned-int int float} 1  nearest) (float 1 linear)
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc>
        void x86ResizeNearestPoint(int inHeight, int inWidth, int inWidthStride, const Tsrc* inData,
                int outHeight, int outWidth, int outWidthStride, Tdst* outData);
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc>
        void x86ResizeLinear(int inHeight, int inWidth, int inWidthStride, const Tsrc* inData,
                int outHeight, int outWidth, int outWidthStride, Tdst* outData);
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc>
        void x86ResizeArea(int inHeight, int inWidth, int inWidthStride, const Tsrc* inData, 
                int outHeight, int outWidth, int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc>
        void armResizeNearestPoint(int inHeight, int inWidth, int inWidthStride, const Tsrc* inData,
                int outHeight, int outWidth, int outWidthStride, Tdst* outData);
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc>
        void armResizeLinear(int inHeight, int inWidth, int inWidthStride, const Tsrc* inData,
                int outHeight, int outWidth, int outWidthStride, Tdst* outData);
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc>
        void armResizeArea(int inHeight, int inWidth, int inWidthStride, const
                Tsrc* inData, int outHeight, int outWidth, int outWidthStride,
                Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc>
        void cudaResizeNearestPoint(int device, cudaStream_t stream,
                int inHeight, int inWidth, int inWidthStride, const Tsrc* inData,
                int outHeight, int outWidth, int outWidthStride, Tdst* outData);
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc>
        void cudaResizeLinear(int device, cudaStream_t stream,
                int inHeight, int inWidth, int inWidthStride, const Tsrc* inData,
                int outHeight, int outWidth, int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc>
        void oclResizeNearestPoint(cl_command_queue queue,
                int inHeight, int inWidth, int inWidthStride, const cl_mem inData,
                int outHeight, int outWidth, int outWidthStride, cl_mem outData);
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc>
        void oclResizeLinear(cl_command_queue queue,
                int inHeight, int inWidth, int inWidthStride, const cl_mem inData,
                int outHeight, int outWidth, int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
    /**
     * @ image warpAffine
     * watch out the element type and the interpolation method of different
     * architecture
     *
     * @param inHeight          input Data's height
     * @param inWidth           input Data's width need to be processed
     * @param inData            input Data
     * @param outHeight         output Data's height
     * @param outWidth          output Data's width need to be processed
     * @param outData           output Data
     * @param affineMatrix      transform matix (Size: 2x3)
     * @param mode              linear interpolation will be used if mode is INTERPOLATION_TYPE_LINEAR, nearest interpolation will be used if mode is INTERPOLATION_TYPE_NEAREST_POINT
     *
     * available now (type channels mode)
     * x86:
     * arm:(uchar {1,4} nearest) (uchar {1,4} linear)
     * cuda:
     **/
#if defined(USE_X86)
    template<typename T, int ncSrc, int ncDst, int nc> void x86WarpAffineNearestPoint(
            int inHeight, int inWidth, int inWidthStride, const T* inData,
            int outHeight, int outWidth, int outWidthStride, T* outData,
            const float* affineMatrix);
    template<typename T, int ncSrc, int ncDst, int nc> void x86WarpAffineLinear(
            int inHeight, int inWidth, int inWidthStride, const T* inData,
            int outHeight, int outWidth, int outWidthStride, T* outData,
            const float* affineMatrix);
#endif
#if defined(USE_ARM)
    template<typename T, int ncSrc, int ncDst, int nc> void armWarpAffineNearestPoint(
            int inHeight, int inWidth, int inWidthStride, const T* inData,
            int outHeight, int outWidth, int outWidthStride, T* outData,
            const float* affineMatrix);
    template<typename T, int ncSrc, int ncDst, int nc> void armWarpAffineLinear(
            int inHeight, int inWidth, int inWidthStride, const T* inData,
            int outHeight, int outWidth, int outWidthStride, T* outData,
            const float* affineMatrix);
#endif
#if defined(USE_CUDA)
    template<typename T, int ncSrc, int ncDst, int nc> void cudaWarpAffineNearestPoint(
            int device, cudaStream_t stream,
            int inHeight, int inWidth, int inWidthStride, const T* inData,
            int outHeight, int outWidth, int outWidthStride, T* outData,
            const float* affineMatrix);
    template<typename T, int ncSrc, int ncDst, int nc> void cudaWarpAffineLinear(
            int device, cudaStream_t stream,
            int inHeight, int inWidth, int inWidthStride, const T* inData,
            int outHeight, int outWidth, int outWidthStride, T* outData,
            const float* affineMatrix);
#endif
#if defined(USE_OCL)
    template<typename T, int ncSrc, int ncDst, int nc>
		void oclWarpAffineNearestPoint(
            cl_command_queue queue,
            int inHeight, int inWidth, int inWidthStride, const cl_mem inData,
            int outHeight, int outWidth, int outWidthStride, cl_mem outData,
            const float* affineMatrix);
    template<typename T, int ncSrc, int ncDst, int nc>
		void oclWarpAffineLinear(
            cl_command_queue queue,
            int inHeight, int inWidth, int inWidthStride, const cl_mem inData,
            int outHeight, int outWidth, int outWidthStride, cl_mem outData,
            const float* affineMatrix);
#endif
    //@}

   //@{
    /**
     * @ image　Remap
     * watch out the element type and the interpolation method of different
     * architecture
     *
     * @param inHeight          input Data's height
     * @param inWidth           input Data's width need to be processed
     * @param inData            input Data
     * @param outHeight         output Data's height
     * @param outWidth          output Data's width need to be processed
     * @param outData           output Data
     * @param mapx     			x matix
     * @param mapx     			y matix
     * @param mode              linear interpolation will be used if mode is INTERPOLATION_TYPE_LINEAR, nearest interpolation will be used if mode is INTERPOLATION_TYPE_NEAREST_POINT
     *
     * available now (type channels mode)
     * x86:
     * arm:(uchar {1,4} nearest) (uchar {1,4} linear)
     * cuda:
     **/
#if defined(USE_X86)
    template<typename T, int ncSrc, int ncDst, int nc> void x86RemapNearestPoint(
            int inHeight, int inWidth, int inWidthStride, const T* inData,
            int outHeight, int outWidth, int outWidthStride, T* outData,
            const float* mapx,const float* mapy);
    template<typename T, int ncSrc, int ncDst, int nc> void x86RemapLinear(
            int inHeight, int inWidth, int inWidthStride, const T* inData,
            int outHeight, int outWidth, int outWidthStride, T* outData,
            const float* mapx,const float* mapy);
#endif
#if defined(USE_ARM)
    template<typename T, int ncSrc, int ncDst, int nc> void armRemapNearestPoint(
            int inHeight, int inWidth, int inWidthStride, const T* inData,
            int outHeight, int outWidth, int outWidthStride, T* outData,
            const float* mapx,const float* mapy);
    template<typename T, int ncSrc, int ncDst, int nc> void armRemapLinear(
            int inHeight, int inWidth, int inWidthStride, const T* inData,
            int outHeight, int outWidth, int outWidthStride, T* outData,
            const float* mapx,const float* mapy);
#endif
#if defined(USE_CUDA)
    template<typename T, int ncSrc, int ncDst, int nc> void cudaRemapNearestPoint(
            int device, cudaStream_t stream,
            int inHeight, int inWidth, int inWidthStride, const T* inData,
            int outHeight, int outWidth, int outWidthStride, T* outData,
            const float* mapx,const float* mapy);
    template<typename T, int ncSrc, int ncDst, int nc> void cudaRemapLinear(
            int device, cudaStream_t stream,
            int inHeight, int inWidth, int inWidthStride, const T* inData,
            int outHeight, int outWidth, int outWidthStride, T* outData,
            const float* mapx,const float* mapy);
#endif
#if defined(USE_OCL)
    template<typename T, int ncSrc, int ncDst, int nc>
		void oclRemapNearestPoint(
            cl_command_queue queue,
            int inHeight, int inWidth, int inWidthStride, const cl_mem inData,
            int outHeight, int outWidth, int outWidthStride, cl_mem outData,
           	cl_mem mapx,cl_mem mapy);
    template<typename T, int ncSrc, int ncDst, int nc>
		void oclRemapLinear(
            cl_command_queue queue,
            int inHeight, int inWidth, int inWidthStride, const cl_mem inData,
            int outHeight, int outWidth, int outWidthStride, cl_mem outData,
            cl_mem mapx,cl_mem mapy);
#endif
    //@}


    //@{
    /**
     * @ image warpPerspective
     * watch out the element type and the interpolation method of different
     * architecture
     *
     * @param inHeight          input Data's height
     * @param inWidth           input Data's width need to be processed
     * @param inData            input Data
     * @param outHeight         output Data's height
     * @param outWidth          output Data's width need to be processed
     * @param outData           output Data
     * @param Perspective matrix    transform matix (Size: 3x3)
     * @param mode              linear interpolation will be used if mode is INTERPOLATION_TYPE_LINEAR, nearest interpolation will be used if mode is INTERPOLATION_TYPE_NEAREST_POINT
     *
     * available now (type channels mode)
     * x86:
     * arm:(uchar {1,4} nearest) (uchar {1,4} linear)
     * cuda:
     **/
#if defined(USE_X86)
#endif
#if defined(USE_ARM)
#endif
#if defined(USE_CUDA)
    template<typename T, int C>
        void cudaWarpPerspective(int device, cudaStream_t stream, int inHeight, int inWidth, int inWidthStride, const T* inData, int outHeight, int outWidth, int outWidthStride, T* outData, const float* perspectiveMatrix);
#endif
#if defined(USE_OCL)
#endif
    //@}

    //@{
    /**
     * @ image transpose
     * watch out the element type
     *
     * @param inHeight          input Data's height
     * @param inWidth           input Data's width need to be processed
     * @param inData            input Data
     * @param outData           output Data
     * @param degree_div90      0<-->No transpose, 1<--> right transpose,
     *                          2<-->vertical transpose, 3<--> left
     *                          transpose
     *
     * available now (type channels mode)
     * x86:
     * arm:
     * cuda: (uchar {1,3,4}) (float {1,3,4})
     **/
#if defined(USE_X86)
#endif
#if defined(USE_ARM)
#endif
#if defined(USE_CUDA)
    template<typename T, int C>
        void cudaImageTranspose(int device, cudaStream_t stream, const int inHeight, const int inWidth, const T* inData, T* outData, const int degree_div90);
#endif
#if defined(USE_OCL)
#endif

//@{
    /**
     * @rotate image or data into the dest size of outDesc
     * watch out the element type and the rotate degree of different
     * architecture
     *
     * @param inHeight          input Data's height
     * @param inWidth           input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outHeight         output Data's height
     * @param outWidth          output Data's width need to be processed
     * @param outWidthStride    output Data's width
     * @param outData           output Data
     * @param degree            the rotate degree, only support 90, 180, 270
     *
     * available now (type channels mode)
     * x86:(uchar 1 linear) (uchar 4 linear)
     * arm:(ucahr {1,3,4} linear) (float {1,3,4} linear)
     * cuda:({uchar short-int unsigned-int int float} 1  nearest) (float 1 linear)
     **/

#if defined(USE_X86)
    template<typename T, int ncSrc, int ncDst, int nc>
        void x86RotateNx90degree(int inHeight, int inWidth, int inWidthStride, const T* inData,
                int outHeight, int outWidth, int outWidthStride, T* outData, int degree);
#endif
#if defined(USE_ARM)
    template<typename T, int ncSrc, int ncDst, int nc>
        void armRotateNx90degree(int inHeight, int inWidth, int inWidthStride, const T* inData,
                int outHeight, int outWidth, int outWidthStride, T* outData, int degree);
#endif
#if defined(USE_CUDA)
    template<typename T, int ncSrc, int ncDst, int nc>
        void cudaRotateNx90degree(int device, cudaStream_t stream,
                int inHeight, int inWidth, int inWidthStride, const T* inData,
                int outHeight, int outWidth, int outWidthStride, T* outData, int degree);
#endif
#if defined(USE_OCL)
    template<typename T, int ncSrc, int ncDst, int nc>
        void oclRotateNx90degree(cl_command_queue queue,
                int inHeight, int inWidth, int inWidthStride, const cl_mem inData,
                int outHeight, int outWidth, int outWidthStride, cl_mem outData, int degree);
#endif
    //@}

//@{
    /**
     * @rotate YUV420 image or data into the dest size of outDesc
     * watch out the element type and the rotate degree of different
     * architecture
     *
     * @param inHeight          input Data's height
     * @param inWidth           input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param outHeight         output Data's height
     * @param outWidth          output Data's width need to be processed
     * @param outWidthStride    output Data's width
     * @param outData           output Data
     * @param degree            the rotate degree, only support 90, 180, 270
     * @param YUV420_type       the YUV420_type, when it is 0, it refer to NV12 or NV21,
     *                          when it is 1, it refer to I420, unsupport other value and type now
     *
     * available now (type channels mode)
     * x86:(uchar 1 linear) (uchar 4 linear)
     * arm:(ucahr {1,3,4} linear) (float {1,3,4} linear)
     * cuda:({uchar short-int unsigned-int int float} 1  nearest) (float 1 linear)
     **/

#if defined(USE_X86)
    template<typename T>
        void x86RotateNx90degree_YUV420(int inHeight, int inWidth, int inWidthStride, const T* inData,
                int outHeight, int outWidth, int outWidthStride, T* outData, int degree, int yuv420_type);
#endif
#if defined(USE_ARM)
    template<typename T>
        void armRotateNx90degree_YUV420(int inHeight, int inWidth, int inWidthStride, const T* inData,
                int outHeight, int outWidth, int outWidthStride, T* outData, int degree, int yuv420_type);
#endif
#if defined(USE_CUDA)
    template<typename T>
        void cudaRotateNx90degree_YUV420(int device, cudaStream_t stream,
                int inHeight, int inWidth, int inWidthStride, const T* inData,
                int outHeight, int outWidth, int outWidthStride, T* outData, int degree, int YUV420_type);
#endif
#if defined(USE_OCL)
    template<typename T>
        void oclRotateNx90degree_YUV420(cl_command_queue queue,
                int inHeight, int inWidth, int inWidthStride, const cl_mem inData,
                int outHeight, int outWidth, int outWidthStride, cl_mem outData, int degree, int YUV420_type);
#endif
    //@}



    //@{
    /**
     * @ Crop an image
     * watch out the element type
     *
     * @param inHeight          input Data's height
     * @param inWidth           input Data's width
     * @param inData            input Data
     * @param left
     * @param top
     * @param outHeight         the cropped image's height
     * @param outWidth          the cropped image's width
     * @param outData           output(cropped) Data
     *
     * available now (type channels mode)
     * x86:
     * arm:
     * cuda: (uchar {1,3,4}) (float {1,3,4})
     **/
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc>
        void x86ImageCrop(
                int p_y, int p_x, int inWidthStride, const Tsrc* inData,
                int outHeight, int outWidth, int outWidthStride, Tdst* outData,
                float ratio);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc>
        void armImageCrop(
                int p_y, int p_x, int inWidthStride, const Tsrc* inData,
                int outHeight, int outWidth, int outWidthStride, Tdst* outData,
                float ratio);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc>
        void cudaImageCrop(int device, cudaStream_t stream,
                int p_y, int p_x, int inWidthStride, const Tsrc* inData,
                int outHeight, int outWidth, int outWidthStride, Tdst* outData,
                float ratio);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc>
        void oclImageCrop(cl_command_queue queue,
                int p_y, int p_x, int inWidthStride, const cl_mem inData,
                int outHeight, int outWidth, int outWidthStride, cl_mem outData,
                float ratio);
#endif
    //@}

    //@{
    /**
     * @filter an image with a general 2D kernel
     *
     * *
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param kernel_len        the length of mask
     * @param outData           output Data
     *
     * available now(type kernel_len sigma)
     * x86:none
     * arm:
     * cuda: none
     **/
#if defined(USE_X86)
    template<typename T, int ncSrc, int ncDst, int nc>
        void x86Filter2DReflect101(int height, int width, int inWidthStride,
                const T* inData, int kernel_len, const float* kernel,
                int outWidthStride, T* outData);
#endif
#if defined(USE_ARM)
    template<typename T, int ncSrc, int ncDst, int nc>
        void armFilter2DReflect101(int height, int width, int inWidthStride,
                const T* inData, int kernel_len, const float* kernel,
                int outWidthStride, T* outData);
#endif
#if defined(USE_CUDA)
    template<typename T, int ncSrc, int ncDst, int nc>
        void cudaFilter2DReflect101(int device, cudaStream_t stream,
				int height, int width, int inWidthStride,
                const T* inData, int kernel_len, const float* kernel,
                int outWidthStride, T* outData);
#endif
#if defined(USE_OCL)
    template<typename T, int ncSrc, int ncDst, int nc>
        void oclFilter2DReflect101(cl_command_queue queue,
				int height, int width, int inWidthStride,
                const cl_mem inData, int kernel_len, const float* ptr,
                int outWidthStride, cl_mem outData);
#endif
    //@}


    //@{
    /**
     * @denoise or obscure an image with gaussian alogrithm
     * arm: accelerated with neon, only 5*5 and 7*7 are supported now
     * only arm version is available
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param kernel_len        the length of mask
     * @param sigma             standard deviation
     * @param outData           output Data
     *
     * available now(type kernel_len sigma)
     * x86:none
     * arm:(uchar 7 2) (uchar 5 2)
     * cuda: none
     **/
#if defined(USE_X86)
    template<typename T, int ncSrc, int ncDst, int nc>
        void x86GaussianBlur(int height, int width, int inWidthStride, const T* inData,
                int kernel_len, float sigma,
                int outWidthStride, T* outData);
#endif
#if defined(USE_ARM)
    template<typename T, int ncSrc, int ncDst, int nc>
        void armGaussianBlur(int height, int width, int inWidthStride, const T* inData,
                int kernel_len, float sigma,
                int outWidthStride, T* outData);
#endif
#if defined(USE_CUDA)

    template<typename T, int ncSrc, int ncDst, int nc>
        void cudaGaussianBlur(int device, cudaStream_t stream,
                int height, int width, int inWidthStride, const T* inData,
                int kernel_len, float sigma,
                int outWidthStride, T* outData);
#endif
#if defined(USE_OCL)
    template<typename T, int ncSrc, int ncDst, int nc>
        void oclGaussianBlur(cl_command_queue queue,
                int height, int width, int inWidthStride, const cl_mem inData,
                int kernel_len, float sigma,
                int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
    /**
     * @blur an image with box kernel
     *
     * *
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param kernel_len        the length of mask
     * @param sigma             standard deviation
     * @param outData           output Data
     *
     * available now(type kernel_len sigma)
     * x86:none
     * arm:(uchar 7 2) (uchar 5 2)
     * cuda: none
     **/
#if defined(USE_X86)
    template<typename T, int ncSrc, int ncDst, int nc>
        void x86boxFilterReflect101(int height, int width, int inWidthStride,
                const T* inData, int kernelx_len, int kernely_len,
                bool normalize, int outWidthStride, T* outData);
#endif
#if defined(USE_ARM)
    template<typename T, int ncSrc, int ncDst, int nc>
        void armboxFilterReflect101(int height, int width, int inWidthStride,
                const T* inData, int kernelx_len, int kernely_len,
                bool normalize, int outWidthStride, T* outData);
#endif
#if defined(USE_CUDA)
    template<typename T, int ncSrc, int ncDst, int nc>
        void cudaboxFilterReflect101(int device, cudaStream_t stream,
				int height, int width, int inWidthStride,
                const T* inData, int kernelx_len, int kernely_len,
                bool normalize, int outWidthStride, T* outData);
#endif
#if defined(USE_OCL)
    template<typename T, int ncSrc, int ncDst, int nc>
        void oclboxFilterReflect101(cl_command_queue command_queue,
				int height, int width, int inWidthStride,
                const cl_mem inData, int kernelx_len, int kernely_len,
                bool normalize, int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
#if defined(USE_X86)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc>
        void x86IntegralImage(int height, int width, int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_ARM)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc>
        void armIntegralImage(int height, int width, int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_CUDA)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc>
        void cudaIntegralImage(int device, cudaStream_t stream,
                int height, int width, int inWidthStride, const Tsrc* inData,
                int outWidthStride, Tdst* outData);
#endif
#if defined(USE_OCL)
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc>
        void oclIntegralImage(cl_command_queue queue,
                int height, int width, int inWidthStride, const cl_mem inData,
                int outWidthStride, cl_mem outData);
#endif
    //@}

    //@{
    /**
     * @arithmetic computation between Matrixes
     * arm: accelerated with neon
     * only arm version is available
     * looks like geam
     *
     * @param inData1           input Data1, it is also the output Data
     * @param inDesc2           layout Descriptor of inData2
     * @param inData2           input Data2
     * @param type              "MATRIX_ARITH_ADD" matrix_add  "MATRIX_ARITH_SUB" matrix_sub "MATRIX_ARITH_MUL" matrix_mul
     **/
    typedef enum {
        MATRIX_ARITH_ADD = 0,
        MATRIX_ARITH_SUB = 1,
        MATRIX_ARITH_MUL = 2,
    } MatrixArithType;

#if defined(USE_X86)
    template<typename T>
        void x86MatrixArith(int height, int width, int inWidthStride, T* inData1, T* inData2, int outWidthStride, T* outData, MatrixArithType type);
#endif
#if defined(USE_ARM)
    template<typename T>
        void armMatrixArith(int height, int width, int inWidthStride, T* inData1, T* inData2, int outWidthStride, T* outData, MatrixArithType type);
#endif
#if defined(USE_CUDA)
    template<typename T>
        void cudaMatrixArith(int device, cudaStream_t stream, int height, int width, int inWidthStride, T* inData1, T* inData2, int outWidthStride, T* outData, MatrixArithType type);
#endif
    //@}

    //@{
    /**
     * @set the limit of the data, the data will be set to a certain value if it
     * is out of limit
     * arm: accelerated with neon
     * only arm version is available
     *
     * @param handle
     * @param inDesc            layout Descriptor of inData, it is matrix for image
     * @param inData            input Data
     * @param value             limit value
     * @param type              "0" set the lower limit  "1" set the upper limit
     **/
    typedef enum {
        MATRIX_CLAMP_MIN = 0,
        MATRIX_CLAMP_MAX = 1,
    } MatrixClampType;
#if defined(USE_X86)
    template<typename T>
        void x86MatrixClamp(int height, int width, int inWidthStride, T* inData, T value, int outWidthStride, T* outData, MatrixClampType type);
#endif
#if defined(USE_ARM)
    template<typename T>
        void armMatrixClamp(int height, int width, int inWidthStride, T* inData, T value, int outWidthStride, T* outData, MatrixClampType type);
#endif
#if defined(USE_CUDA)
    template<typename T>
        void cudaMatrixClamp(int device, cudaStream_t stream, int height, int width, int inWidthStride, T* inData, T value, int outWidthStride, T* outData, MatrixClampType type);
#endif
    //@}


    //@{
    /**
     * @blur an image with bilateral filter
     *
     * *
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param color             range_sigma
     * @param space             space_sigma
     * @param outWidthStride    out Data's width
     * @param outData           output Data
     *
     * available now(type inChannel outChannel processedChannels)
     * x86:none
     * arm:(float 1 1 1) (float 3 3 3) (float 4 4 4 4)
     * cuda: none
     **/
#if defined(USE_X86)
    template<typename T, int ncSrc, int ncDst, int nc>
        void x86BilateralFilter(int height, int width, int inWidthStride,
                const T* inData, int diameter,
                double color, double space, int outWidthStride, T* outData);
#endif
#if defined(USE_ARM)
    template<typename T, int ncSrc, int ncDst, int nc>
        void armBilateralFilter(int height, int width, int inWidthStride,
                const T* inData, int diameter,
                double color, double space, int outWidthStride, T* outData);
#endif
#if defined(USE_CUDA)
    template<typename T, int ncSrc, int ncDst, int nc>
        void cudaBilateralFilter(int device, cudaStream_t stream,
				int height, int width, int inWidthStride,
                const T* inData, int diameter,
                double color, double space, int outWidthStride, T* outData);
#endif
#if defined(USE_OCL)
    template<typename T, int ncSrc, int ncDst, int nc>
        void oclBilateralFilter(cl_command_queue command_queue,
				int height, int width, int inWidthStride,
                const cl_mem inData, int diameter,
                double color, double space, int outWidthStride, cl_mem outData);
#endif
    //@}


    //@{
    /**
     * @blur an image with fast bilateral filter
     *
     * *
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param baseWidthStride   base Data's width
     * @param baseData          base Data
     * @param space_sigma       space_sigma
     * @param range_sigma       range_sigma
     * @param outWidthStride    out Data's width
     * @param outData           output Data
     *
     * available now(type inChannel baseChannel outChannel)
     * x86:none
     * arm:(float 1 1 1) (float 3 3 3) (float 4 4 4 4)
     * cuda: none
     **/
#if defined(USE_X86)
    template<typename T, int ncSrc, int ncBase, int ncDst, int nc>
        void x86FastBilateralFilter(int height, int width, int inWidthStride,
                const T* inData, int baseWidthStride, const T* baseData,
                double range_sigma, double space_sigma, int outWidthStride, T* outData);
#endif
#if defined(USE_ARM)
    template<typename T, int ncSrc, int ncBase, int ncDst, int nc>
        void armFastBilateralFilter(int height, int width, int inWidthStride,
                const T* inData, int baseWidthStride, const T* baseData,
                double range_sigma, double space_sigma, int outWidthStride, T* outData);
#endif
#if defined(USE_CUDA)
    template<typename T, int ncSrc, int ncBase, int ncDst, int nc>
        void cudaFastBilateralFilter(int height, int width, int inWidthStride,
                const T* inData, int baseWidthStride, const T* baseData,
                double range_sigma, double space_sigma, int outWidthStride, T* outData);
#endif
#if defined(USE_OCL)
    template<typename T, int ncSrc, int ncBase, int ncDst, int nc>
        void oclFastBilateralFilter(int height, int width, int inWidthStride,
                const T* inData, int baseWidthStride, const T* baseData,
                double range_sigma, double space_sigma, int outWidthStride, T* outData);
#endif
    //@}

        //@{
    /**
     * @erode an image 
     *
     * *
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param kernelx_len       filter's height
     * @param kernely_len       filter's width
     * @param outWidthStride    out Data's width
     * @param outData           output Data
     * @param cn                number of channel(s)
     *
     * available now(type inChannel baseChannel outChannel)
     * x86:(uchar/float 1 1 1) (uchar/float 3 3 3) (uchar/float 4 4 4)
     * arm:(uchar/float 1 1 1) (uchar/float 3 3 3) (uchar/float 4 4 4)
     * cuda: none
     **/
#if defined(USE_X86)
     template<typename T, int ncSrc, int ncBase, int ncDst>
        void x86minFilter(int height, int width, int inWidthStride, 
            const T* inData, int kernelx_len, int kernely_len, 
            const unsigned char* element, int outWidthStride, T* outData);
#endif
#if defined(USE_ARM)
     template<typename T, int ncSrc, int ncBase, int ncDst>
        void armminFilter(int height, int width, int inWidthStride, 
            const T* inData, int kernelx_len, int kernely_len, 
            const unsigned char* element, int outWidthStride, T* outData);
#endif
    //@}

    //@{
    /**
     * @dilate an image 
     *
     * *
     *
     * @param height            input Data's height
     * @param width             input Data's width need to be processed
     * @param inWidthStride     input Data's width
     * @param inData            input Data
     * @param kernelx_len       filter's height
     * @param kernely_len       filter's width
     * @param outWidthStride    out Data's width
     * @param outData           output Data
     * @param cn                number of channel(s)
     *
     * available now(type inChannel baseChannel outChannel)
     * x86:(uchar/float 1 1 1) (uchar/float 3 3 3) (uchar/float 4 4 4)
     * arm:(uchar/float 1 1 1) (uchar/float 3 3 3) (uchar/float 4 4 4)
     * cuda: none
     **/
#if defined(USE_X86)
     template<typename T, int ncSrc, int ncBase, int ncDst>
        void x86maxFilter(int height, int width, int inWidthStride, 
            const T* inData, int kernelx_len, int kernely_len, 
            const unsigned char* element, int outWidthStride, T* outData);
#endif
#if defined(USE_ARM)
     template<typename T, int ncSrc, int ncBase, int ncDst>
        void armmaxFilter(int height, int width, int inWidthStride, 
            const T* inData, int kernelx_len, int kernely_len, 
            const unsigned char* element, int outWidthStride, T* outData);
#endif
    //@}
} };

#endif
