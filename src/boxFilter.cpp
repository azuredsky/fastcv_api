#include <fastcv.hpp>
#include <image.h>
#include<stdio.h>

namespace HPC { namespace fastcv {
/*
    typedef enum {
        BORDER_CONSTANT = 0, //iiiiii|abcdefgh|iiiiiii with some specified i
        BORDER_REPLICATE = 1, // aaaaaa|abcdefgh|hhhhhhh
        BORDER_REFLECT = 2, // fedcba|abcdefgh|hgfedcb
        BORDER_WRAP    = 3, // cdefgh|abcdefgh|abcdefg
        BORDER_REFLECT101 = 4, // gfedcb|abcdefgh|gfedcba
        BORDER_DEFAULT = 4, // same as BORDER_REFLECT_101
        BORDER_TRANSPARENT = 5, // uvwxyz|absdefgh|i
        BORDER_ISOLATED = 6, //do not look outside of ROI
    } BorderType;
*/

    template<BorderType bt, typename T, int ncSrc, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t boxFilter(const Mat<T, ncSrc, type>& src, Size2i ks, Point2i anchor, Mat<T, ncDst, type> *dst, bool normalized)
        {

#if __cplusplus >= 201103L
#endif
            if(NULL == dst->ptr()) return HPC_POINTER_NULL;
            if(NULL == src.ptr()) return HPC_POINTER_NULL;
            if(src.numDimensions() == 0 || dst->numDimensions() == 0) return HPC_NOT_INITIALIZED;

            size_t sh = src.height();
            size_t sw = src.width();
            int sstep = src.widthStep()/sizeof(T);
           // T* sp = src.ptr();
            //size_t dh = dst->height();
            //size_t dw = dst->width();
            int dstep = dst->widthStep()/sizeof(T);
          //  T* dp = dst->ptr();
            int kernelx_len = ks.w();
            int kernely_len = ks.h();

            void* sp = src.ptr();
            void* dp = dst->ptr();

            if(src.width() != dst->width() || src.height() != dst->height()) return HPC_INVALID_ARGS;

            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                x86boxFilterReflect101<T, ncSrc, ncDst, nc>(sh, sw, sstep, (const T*)sp,
                    kernelx_len, kernely_len, normalized, dstep, (T*)dp);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                if( bt == BORDER_DEFAULT && anchor.x == -1 && anchor.y == -1) {
                    armboxFilterReflect101<T, ncSrc, ncDst, nc>(src.height(), src.width(),
                        sstep, (T*)src.ptr(), ks.w(), ks.h(), normalized, dstep,
                        (T*)dst->ptr());
                } else {
                    return HPC_NOT_SUPPORTED;
                }
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                cudaboxFilterReflect101<T,ncSrc,ncDst,ncSrc>(0,0,sh,sw,sstep,(const T*)sp,kernelx_len,kernely_len,normalized,dstep,(T*)dp);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                oclboxFilterReflect101<T,ncSrc,ncDst,ncSrc>(opencl.getCommandQueue(),sh,sw,sstep,(cl_mem)sp,kernelx_len,kernely_len,normalized,dstep,(cl_mem)dp);
#endif
            }

            return HPC_SUCCESS;
        }

#if defined(FASTCV_USE_X86)
    template HPCStatus_t boxFilter<BORDER_DEFAULT, float, 1, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& src, Size2i ks, Point2i anchor, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *dst, bool normalized);
    template HPCStatus_t boxFilter<BORDER_DEFAULT, float, 3, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Size2i ks, Point2i anchor, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst, bool normalized);
    template HPCStatus_t boxFilter<BORDER_DEFAULT, float, 4, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& src, Size2i ks, Point2i anchor, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *dst, bool normalized);

    template HPCStatus_t boxFilter<BORDER_DEFAULT, uchar, 1, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, Size2i ks, Point2i anchor, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst, bool normalized);
    template HPCStatus_t boxFilter<BORDER_DEFAULT, uchar, 3, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Size2i ks, Point2i anchor, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst, bool normalized);
    template HPCStatus_t boxFilter<BORDER_DEFAULT, uchar, 4, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, Size2i ks, Point2i anchor, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst, bool normalized);
#endif

#if defined(FASTCV_USE_ARM)
    template HPCStatus_t boxFilter<BORDER_DEFAULT, float, 1, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& src, Size2i ks, Point2i anchor, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *dst, bool normalized);
    template HPCStatus_t boxFilter<BORDER_DEFAULT, float, 3, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, Size2i ks, Point2i anchor, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst, bool normalized);
    template HPCStatus_t boxFilter<BORDER_DEFAULT, float, 4, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& src, Size2i ks, Point2i anchor, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *dst, bool normalized);
	template HPCStatus_t boxFilter<BORDER_DEFAULT, uchar, 1, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, Size2i ks, Point2i anchor, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst, bool normalized);
    template HPCStatus_t boxFilter<BORDER_DEFAULT, uchar, 3, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Size2i ks, Point2i anchor, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst, bool normalized);
    template HPCStatus_t boxFilter<BORDER_DEFAULT, uchar, 4, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, Size2i ks, Point2i anchor, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst, bool normalized);
#endif

#if defined(FASTCV_USE_CUDA)
    template HPCStatus_t boxFilter<BORDER_DEFAULT, float, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Size2i ks, Point2i anchor, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *dst, bool normalized);
    template HPCStatus_t boxFilter<BORDER_DEFAULT, float, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Size2i ks, Point2i anchor, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst, bool normalized);
    template HPCStatus_t boxFilter<BORDER_DEFAULT, float, 4, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Size2i ks, Point2i anchor, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *dst, bool normalized);

    template HPCStatus_t boxFilter<BORDER_DEFAULT, uchar, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Size2i ks, Point2i anchor, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst, bool normalized);
    template HPCStatus_t boxFilter<BORDER_DEFAULT, uchar, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Size2i ks, Point2i anchor, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst, bool normalized);
    template HPCStatus_t boxFilter<BORDER_DEFAULT, uchar, 4, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Size2i ks, Point2i anchor, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst, bool normalized);

    template HPCStatus_t boxFilter<BORDER_DEFAULT, half_t, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Size2i ks, Point2i anchor, Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA> *dst, bool normalized);
    template HPCStatus_t boxFilter<BORDER_DEFAULT, half_t, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Size2i ks, Point2i anchor, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst, bool normalized);
    template HPCStatus_t boxFilter<BORDER_DEFAULT, half_t, 4, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Size2i ks, Point2i anchor, Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA> *dst, bool normalized);
#endif

#if defined(FASTCV_USE_OCL)
    template HPCStatus_t boxFilter<BORDER_DEFAULT, float, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& src, Size2i ks, Point2i anchor, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *dst, bool normalized);
    template HPCStatus_t boxFilter<BORDER_DEFAULT, float, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Size2i ks, Point2i anchor, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst, bool normalized);
    template HPCStatus_t boxFilter<BORDER_DEFAULT, float, 4, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& src, Size2i ks, Point2i anchor, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *dst, bool normalized);

    template HPCStatus_t boxFilter<BORDER_DEFAULT, uchar, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, Size2i ks, Point2i anchor, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst, bool normalized);
    template HPCStatus_t boxFilter<BORDER_DEFAULT, uchar, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Size2i ks, Point2i anchor, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst, bool normalized);
    template HPCStatus_t boxFilter<BORDER_DEFAULT, uchar, 4, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, Size2i ks, Point2i anchor, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst, bool normalized);

    template HPCStatus_t boxFilter<BORDER_DEFAULT, cl_half, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL>& src, Size2i ks, Point2i anchor, Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL> *dst, bool normalized);
    template HPCStatus_t boxFilter<BORDER_DEFAULT, cl_half, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, Size2i ks, Point2i anchor, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst, bool normalized);
    template HPCStatus_t boxFilter<BORDER_DEFAULT, cl_half, 4, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL>& src, Size2i ks, Point2i anchor, Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL> *dst, bool normalized);
#endif
}}
