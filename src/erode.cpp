#include <fastcv.hpp>
#include <image.h>
#include<stdio.h>

namespace HPC { namespace fastcv {


    template<typename T, int ncSrc, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t erode(const Mat<T, ncSrc, type>& src, const Mat<uchar, 1, type>& element, Mat<T, ncDst, type> *dst)
        {

            if(NULL == dst) return HPC_POINTER_NULL;
            if(src.numDimensions() == 0 || element.numDimensions() == 0 || dst->numDimensions() == 0) return HPC_NOT_INITIALIZED;

            size_t sh = src.height();
            size_t sw = src.width();
            int sstep = src.widthStep()/sizeof(T);
            int dstep = dst->widthStep()/sizeof(T);

            int kernelx_len = element.height();
            int kernely_len = element.width();

            void* sp = src.ptr();
            void* ep = element.ptr();
            void* dp = dst->ptr();

            if(src.width() != dst->width() || src.height() != dst->height()) return HPC_INVALID_ARGS;

            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                x86minFilter<T, ncSrc, ncDst, ncSrc>(sh, sw, sstep, (const T*)sp, 
                    kernelx_len, kernely_len, (const uchar*) ep, dstep, (T*)dp);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                armminFilter<T, ncSrc, ncDst, ncSrc>(sh, sw, sstep, (const T*)sp, 
                    kernelx_len, kernely_len, (const uchar*) ep, dstep, (T*)dp);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                return HPC_NOT_SUPPORTED;
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
               return HPC_NOT_SUPPORTED;
#endif
            }

            return HPC_SUCCESS;
        }

#if defined(FASTCV_USE_X86)
    template HPCStatus_t erode<uchar, 1, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& element, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t erode<uchar, 3, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& element, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t erode<uchar, 4, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& element, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst);

    template HPCStatus_t erode<float, 1, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& src, const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& element, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *dst);    
    template HPCStatus_t erode<float, 3, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& element, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t erode<float, 4, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& src, const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& element, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *dst);      
#endif

#if defined(FASTCV_USE_ARM)
    template HPCStatus_t erode<uchar, 1, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& element, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t erode<uchar, 3, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& element, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t erode<uchar, 4, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& element, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst);

    template HPCStatus_t erode<float, 1, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& src, const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& element, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t erode<float, 3, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& element, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t erode<float, 4, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& src, const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& element, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *dst);            

#endif

#if defined(FASTCV_USE_CUDA)
#endif

#if defined(FASTCV_USE_OCL)
#endif
}}
