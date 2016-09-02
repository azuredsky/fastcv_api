#include <fastcv.hpp>
#include<image.h>
namespace HPC { namespace fastcv {
    template<typename T, int filterSize, int ncSrc, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t gaussianBlur(const Mat<T, ncSrc, type>& src, Mat<T, ncDst, type> *dst) {
#if __cplusplus >= 201103L
            static_assert(nc<=ncSrc && nc<=ncDst, CHECK_CH_MACRO);
#endif
            if(src.ptr() == NULL) return HPC_POINTER_NULL;
            if(dst->ptr() == NULL) return HPC_POINTER_NULL;

            if(NULL == dst) return HPC_INVALID_ARGS;
            if(src.numDimensions() == 0 || dst->numDimensions() == 0) return HPC_NOT_INITIALIZED;
            int sw = src.width();
            int sh = src.height();

            int sstep = src.widthStep()/sizeof(T);
            //  const T* sp = src.ptr();
            int dw = dst->width();
            int dh = dst->height();
            int dstep = dst->widthStep()/sizeof(T);

            //  T* dp = dst->ptr();
            void* sp = src.ptr();
            void* dp = dst->ptr();

            if(sw != dw || sh != dh) return HPC_INVALID_ARGS;

            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                x86GaussianBlur<T,ncSrc,ncDst,nc>(sh, sw, sstep, (const T*)sp,
                    filterSize, -1, dstep, (T*)dp);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                armGaussianBlur<T,ncSrc,ncDst,nc>(sh, sw, sstep, (const T*)sp,
                    filterSize, -1, dstep, (T*)dp);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                cudaGaussianBlur <T,ncSrc,ncDst,nc> (0, 0, sh, sw, sstep, (const T*)sp,
                    filterSize, -1, dstep, (T*)dp);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                oclGaussianBlur <T,ncSrc,ncDst,nc> (opencl.getCommandQueue(), sh, sw, sstep, (cl_mem)sp,
                    filterSize, -1, dstep, (cl_mem)dp);
#endif
            }
            return HPC_SUCCESS;
        }

#if defined(FASTCV_USE_X86)
    template HPCStatus_t gaussianBlur<float, 3, 1, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t gaussianBlur<float, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t gaussianBlur<float, 3, 4, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *dst);

    template HPCStatus_t gaussianBlur<float, 5, 1, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t gaussianBlur<float, 5, 3, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t gaussianBlur<float, 5, 4, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *dst);

    template HPCStatus_t gaussianBlur<float, 7, 1, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t gaussianBlur<float, 7, 3, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t gaussianBlur<float, 7, 4, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *dst);

    template HPCStatus_t gaussianBlur<uchar, 3, 1, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t gaussianBlur<uchar, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t gaussianBlur<uchar, 3, 4, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst);

    template HPCStatus_t gaussianBlur<uchar, 5, 1, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t gaussianBlur<uchar, 5, 3, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t gaussianBlur<uchar, 5, 4, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst);

    template HPCStatus_t gaussianBlur<uchar, 7, 1, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t gaussianBlur<uchar, 7, 3, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t gaussianBlur<uchar, 7, 4, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst);
#endif

#if defined(FASTCV_USE_ARM)
    template HPCStatus_t gaussianBlur<float, 3, 1, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t gaussianBlur<float, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t gaussianBlur<float, 3, 4, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *dst);

    template HPCStatus_t gaussianBlur<float, 5, 1, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t gaussianBlur<float, 5, 3, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t gaussianBlur<float, 5, 4, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *dst);

    template HPCStatus_t gaussianBlur<float, 7, 1, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t gaussianBlur<float, 7, 3, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t gaussianBlur<float, 7, 4, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *dst);

    template HPCStatus_t gaussianBlur<uchar, 3, 1, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t gaussianBlur<uchar, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t gaussianBlur<uchar, 3, 4, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst);

    template HPCStatus_t gaussianBlur<uchar, 5, 1, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t gaussianBlur<uchar, 5, 3, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t gaussianBlur<uchar, 5, 4, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst);

    template HPCStatus_t gaussianBlur<uchar, 7, 1, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t gaussianBlur<uchar, 7, 3, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t gaussianBlur<uchar, 7, 4, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst);
#endif

#if defined(FASTCV_USE_CUDA)
    template HPCStatus_t gaussianBlur<float, 3, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t gaussianBlur<float, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t gaussianBlur<float, 3, 4, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    template HPCStatus_t gaussianBlur<float, 5, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t gaussianBlur<float, 5, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t gaussianBlur<float, 5, 4, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    template HPCStatus_t gaussianBlur<float, 7, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t gaussianBlur<float, 7, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t gaussianBlur<float, 7, 4, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    template HPCStatus_t gaussianBlur<uchar, 3, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t gaussianBlur<uchar, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t gaussianBlur<uchar, 3, 4, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    template HPCStatus_t gaussianBlur<uchar, 5, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t gaussianBlur<uchar, 5, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t gaussianBlur<uchar, 5, 4, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    template HPCStatus_t gaussianBlur<uchar, 7, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t gaussianBlur<uchar, 7, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t gaussianBlur<uchar, 7, 4, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    template HPCStatus_t gaussianBlur<half_t, 3, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t gaussianBlur<half_t, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t gaussianBlur<half_t, 3, 4, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    template HPCStatus_t gaussianBlur<half_t, 5, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t gaussianBlur<half_t, 5, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t gaussianBlur<half_t, 5, 4, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    template HPCStatus_t gaussianBlur<half_t, 7, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t gaussianBlur<half_t, 7, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t gaussianBlur<half_t, 7, 4, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
#endif

#if defined(FASTCV_USE_OCL)
    template HPCStatus_t gaussianBlur<float, 3, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t gaussianBlur<float, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t gaussianBlur<float, 3, 4, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    template HPCStatus_t gaussianBlur<float, 5, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t gaussianBlur<float, 5, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t gaussianBlur<float, 5, 4, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    template HPCStatus_t gaussianBlur<float, 7, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t gaussianBlur<float, 7, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t gaussianBlur<float, 7, 4, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    template HPCStatus_t gaussianBlur<uchar, 3, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t gaussianBlur<uchar, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t gaussianBlur<uchar, 3, 4, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    template HPCStatus_t gaussianBlur<uchar, 5, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t gaussianBlur<uchar, 5, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t gaussianBlur<uchar, 5, 4, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    template HPCStatus_t gaussianBlur<uchar, 7, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t gaussianBlur<uchar, 7, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t gaussianBlur<uchar, 7, 4, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    template HPCStatus_t gaussianBlur<cl_half, 3, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t gaussianBlur<cl_half, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t gaussianBlur<cl_half, 3, 4, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    template HPCStatus_t gaussianBlur<cl_half, 5, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t gaussianBlur<cl_half, 5, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t gaussianBlur<cl_half, 5, 4, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    template HPCStatus_t gaussianBlur<cl_half, 7, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t gaussianBlur<cl_half, 7, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t gaussianBlur<cl_half, 7, 4, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
#endif

}}


