#include <fastcv.hpp>
#include <image.h>
#include<stdio.h>

namespace HPC { namespace fastcv {

    template<BorderType bt, typename T, int ncSrc, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t bilateralFilter(const Mat<T, ncSrc, type>& src, int diameter, double color, double space, Mat<T, ncDst, type> *dst) {
#if __cplusplus >= 201103L
#endif
            if(src.ptr() == NULL) return HPC_POINTER_NULL;
            if(dst->ptr() == NULL) return HPC_POINTER_NULL;
            
            int sstep = src.widthStep()/sizeof(T);
            int dstep = dst->widthStep()/sizeof(T);

            if(src.width() != dst->width() || src.height() != dst->height() ||
                    sstep < src.width()*ncSrc || dstep < dst->width()*ncDst) return HPC_INVALID_ARGS;

            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                if(bt == BORDER_DEFAULT) {
                    x86BilateralFilter<T, ncSrc, ncDst, nc>(src.height(), src.width(),
                        sstep, (T*)src.ptr(), diameter, color, space,
                        dstep, (T*)dst->ptr());
                } else {
                    return HPC_NOT_SUPPORTED;
                }
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                if(bt == BORDER_DEFAULT) {
                    armBilateralFilter<T, ncSrc, ncDst, nc>(src.height(), src.width(),
                        sstep, (T*)src.ptr(), diameter, color, space,
                        dstep, (T*)dst->ptr());
                } else {
                    return HPC_NOT_SUPPORTED;
                }
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                cudaBilateralFilter<T,ncSrc,ncDst,nc>(0, 0, src.height(), src.width(),
                    sstep, (T*)src.ptr(), diameter, color, space,
                    dstep, (T*)dst->ptr());
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                cl_command_queue commandQueue = opencl.getCommandQueue();
                cl_mem inData = (cl_mem)src.ptr();
                cl_mem outData = (cl_mem)dst->ptr();
                oclBilateralFilter<T,ncSrc,ncDst,nc>(commandQueue, src.height(), src.width(),
                    sstep, inData, diameter, color, space,
                    dstep, outData);
#endif
            }
            return HPC_SUCCESS;
        }

#if defined(FASTCV_USE_X86)
    template HPCStatus_t bilateralFilter<BORDER_DEFAULT, uchar, 1, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, int diameter, double color, double space, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t bilateralFilter<BORDER_DEFAULT, uchar, 3, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, int diameter, double color, double space, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t bilateralFilter<BORDER_DEFAULT, float, 1, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& src, int diameter, double color, double space, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t bilateralFilter<BORDER_DEFAULT, float, 3, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, int diameter, double color, double space, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
#endif

#if defined(FASTCV_USE_ARM)
    template HPCStatus_t bilateralFilter<BORDER_DEFAULT, uchar, 1, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, int diameter, double color, double space, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t bilateralFilter<BORDER_DEFAULT, uchar, 3, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, int diameter, double color, double space, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t bilateralFilter<BORDER_DEFAULT, float, 1, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& src, int diameter, double color, double space, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t bilateralFilter<BORDER_DEFAULT, float, 3, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, int diameter, double color, double space, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
#endif

#if defined(FASTCV_USE_CUDA)
    template HPCStatus_t bilateralFilter<BORDER_DEFAULT, uchar, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, int diameter, double color, double space, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t bilateralFilter<BORDER_DEFAULT, uchar, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, int diameter, double color, double space, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t bilateralFilter<BORDER_DEFAULT, float, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& src, int diameter, double color, double space, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t bilateralFilter<BORDER_DEFAULT, float, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, int diameter, double color, double space, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t bilateralFilter<BORDER_DEFAULT, half_t, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA>& src, int diameter, double color, double space, Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t bilateralFilter<BORDER_DEFAULT, half_t, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, int diameter, double color, double space, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
#endif

#if defined(FASTCV_USE_OCL)
    template HPCStatus_t bilateralFilter<BORDER_DEFAULT, uchar, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, int diameter, double color, double space, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t bilateralFilter<BORDER_DEFAULT, uchar, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, int diameter, double color, double space, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t bilateralFilter<BORDER_DEFAULT, float, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& src, int diameter, double color, double space, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t bilateralFilter<BORDER_DEFAULT, float, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, int diameter, double color, double space, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t bilateralFilter<BORDER_DEFAULT, cl_half, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL>& src, int diameter, double color, double space, Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t bilateralFilter<BORDER_DEFAULT, cl_half, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, int diameter, double color, double space, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
#endif
}}
