#include <fastcv.hpp>
#include <image.h>
namespace HPC { namespace fastcv {
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t convertTo(const Mat<Tsrc, ncSrc, type>& src, float ratio, Mat<Tdst, ncDst, type> *dst) {
#if __cplusplus >= 201103L
            static_assert(nc<=ncSrc && nc<=ncDst, CHECK_CH_MACRO);
#endif
            if(src.ptr() == NULL) return HPC_POINTER_NULL;
            if(dst->ptr() == NULL) return HPC_POINTER_NULL;

            if(NULL == dst) return HPC_INVALID_ARGS;
            if(src.numDimensions() == 0 || dst->numDimensions() == 0) return HPC_NOT_INITIALIZED;
            int sw = src.width();
            int sh = src.height();
            int sstep = src.widthStep()/sizeof(Tsrc);

            //const Tsrc* sp = src.ptr();
            int dw = dst->width();
            int dh = dst->height();
            int dstep = dst->widthStep()/sizeof(Tdst);

            //Tdst* dp = dst->ptr();
            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                const Tsrc* sp = (Tsrc*)src.ptr();
                Tdst* dp = (Tdst*)dst->ptr();
                if(sw != dw || sh != dh || sstep < sw*ncSrc || dstep < dw*ncDst) return HPC_INVALID_ARGS;

                x86ChangeDataTypeAndScale<Tsrc, ncSrc, Tdst, ncDst, nc>(
                    sh, sw, sstep, sp,
                    ratio, dstep, dp);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                const Tsrc* sp = (Tsrc*)src.ptr();
                Tdst* dp = (Tdst*)dst->ptr();

                if(sw != dw || sh != dh || sstep < sw*ncSrc || dstep < dw*ncDst) return HPC_INVALID_ARGS;

                armChangeDataTypeAndScale<Tsrc, ncSrc, Tdst, ncDst, nc>(
                    sh, sw, sstep, sp,
                    ratio, dstep, dp);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                const Tsrc* sp = (Tsrc*)src.ptr();
                Tdst* dp = (Tdst*)dst->ptr();

                if(sw != dw || sh != dh || sstep < sw*ncSrc || dstep < dw*ncDst) return HPC_INVALID_ARGS;

                cudaChangeDataTypeAndScale<Tsrc, ncSrc, Tdst, ncDst, nc>(0,0,
                    sh, sw, sstep, sp,
                    ratio, dstep, dp);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                cl_mem sp = (cl_mem)src.ptr();
                cl_mem dp = (cl_mem)dst->ptr();
                cl_command_queue commandQueue= opencl.getCommandQueue();

                if(sw != dw || sh != dh || sstep < sw*ncSrc || dstep < dw*ncDst) return HPC_INVALID_ARGS;

                oclChangeDataTypeAndScale<Tsrc, ncSrc, Tdst, ncDst, nc>(
                    commandQueue,
                    sh, sw, sstep, sp,
                    ratio, dstep, dp);
#endif
            }
            return HPC_SUCCESS;
        }

#if defined(FASTCV_USE_X86)
        template HPCStatus_t convertTo<uchar, 1, float, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, float ratio, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *dst);
        template HPCStatus_t convertTo<uchar, 3, float, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, float ratio, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
        template HPCStatus_t convertTo<uchar, 4, float, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, float ratio, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *dst);
        template HPCStatus_t convertTo<float, 1, uchar, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& src, float ratio, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
        template HPCStatus_t convertTo<float, 3, uchar, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, float ratio, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
        template HPCStatus_t convertTo<float, 4, uchar, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& src, float ratio, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst);
#endif

#if defined(FASTCV_USE_ARM)
        template HPCStatus_t convertTo<uchar, 1, float, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, float ratio, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
        template HPCStatus_t convertTo<uchar, 3, float, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, float ratio, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
        template HPCStatus_t convertTo<uchar, 4, float, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, float ratio, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *dst);
        template HPCStatus_t convertTo<float, 1, uchar, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& src, float ratio, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
        template HPCStatus_t convertTo<float, 3, uchar, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, float ratio, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
        template HPCStatus_t convertTo<float, 4, uchar, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& src, float ratio, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst);
#endif

#if defined(FASTCV_USE_CUDA)
        template HPCStatus_t convertTo<uchar, 1, float, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, float ratio, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t convertTo<uchar, 3, float, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, float ratio, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t convertTo<uchar, 4, float, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, float ratio, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t convertTo<float, 1, uchar, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& src, float ratio, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t convertTo<float, 3, uchar, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, float ratio, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t convertTo<float, 4, uchar, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& src, float ratio, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

        template HPCStatus_t convertTo<uchar, 1, half_t, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, float ratio, Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t convertTo<uchar, 3, half_t, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, float ratio, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t convertTo<uchar, 4, half_t, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, float ratio, Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t convertTo<half_t, 1, uchar, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA>& src, float ratio, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t convertTo<half_t, 3, uchar, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, float ratio, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t convertTo<half_t, 4, uchar, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA>& src, float ratio, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
#endif

#if defined(FASTCV_USE_OCL)
        template HPCStatus_t convertTo<uchar, 1, float, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, float ratio, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t convertTo<uchar, 3, float, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, float ratio, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t convertTo<uchar, 4, float, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, float ratio, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t convertTo<float, 1, uchar, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& src, float ratio, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t convertTo<float, 3, uchar, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, float ratio, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t convertTo<float, 4, uchar, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& src, float ratio, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

        template HPCStatus_t convertTo<uchar, 1, cl_half, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, float ratio, Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t convertTo<uchar, 3, cl_half, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, float ratio, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t convertTo<uchar, 4, cl_half, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, float ratio, Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t convertTo<cl_half, 1, uchar, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL>& src, float ratio, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t convertTo<cl_half, 3, uchar, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, float ratio, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t convertTo<cl_half, 4, uchar, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL>& src, float ratio, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
#endif
}}
