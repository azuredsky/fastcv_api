#include <fastcv.hpp>
#include <image.h>

namespace HPC { namespace fastcv {
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t integral(const Mat<Tsrc, ncSrc, type>& src, Mat<Tdst, ncDst, type> *dst) {
            if(src.ptr() == NULL) return HPC_POINTER_NULL;
            if(dst->ptr() == NULL) return HPC_POINTER_NULL;
            if(src.numDimensions() == 0 || dst->numDimensions() == 0) return HPC_NOT_INITIALIZED;

            size_t sh = src.height();
            size_t sw = src.width();
            int sstep = src.widthStep()/sizeof(Tsrc);
           // Tsrc* sp = src.ptr();
            size_t dh = dst->height();
            size_t dw = dst->width();
            int dstep = dst->widthStep()/sizeof(Tdst);
           // Tdst* dp = dst->ptr();
			void* sp = src.ptr();
			void* dp = dst->ptr();

            if(sw != dw) return HPC_INVALID_ARGS;
            if(sh != dh) return HPC_INVALID_ARGS;
            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                x86IntegralImage<Tsrc, ncSrc, Tdst, ncDst, nc>(sh, sw, sstep, (const Tsrc*)sp, dstep, (Tdst*)dp);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                armIntegralImage<Tsrc, ncSrc, Tdst, ncDst, nc>(sh, sw, sstep, (const Tsrc*)sp, dstep, (Tdst*)dp);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                cudaIntegralImage<Tsrc, ncSrc, Tdst, ncDst, nc>(0, 0, sh, sw, sstep, (const Tsrc*)sp, dstep, (Tdst*)dp);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                oclIntegralImage<Tsrc, ncSrc, Tdst, ncDst, nc>(opencl.getCommandQueue(), sh, sw, sstep, (cl_mem)sp, dstep, (cl_mem)dp);
#endif
            }
            return HPC_SUCCESS;
        }

#if defined(FASTCV_USE_X86)
/*    template HPCStatus_t integral<uchar, 1, int, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<int, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t integral<uchar, 3, int, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<int, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t integral<uchar, 4, int, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<int, 4, EcoEnv_t::ECO_ENV_X86> *dst);
*/
    template HPCStatus_t integral<float, 1, float, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t integral<float, 3, float, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t integral<float, 4, float, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *dst);
#endif

#if defined(FASTCV_USE_ARM)
/*    template HPCStatus_t integral<uchar, 1, int, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<int, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t integral<uchar, 3, int, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<int, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t integral<uchar, 4, int, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<int, 4, EcoEnv_t::ECO_ENV_ARM> *dst);
*/
    template HPCStatus_t integral<float, 1, float, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t integral<float, 3, float, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t integral<float, 4, float, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *dst);
#endif

#if defined(FASTCV_USE_CUDA)
    template HPCStatus_t integral<uchar, 1, int, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<int, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t integral<uchar, 3, int, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<int, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t integral<uchar, 4, int, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<int, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    template HPCStatus_t integral<float, 1, float, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t integral<float, 3, float, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t integral<float, 4, float, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    template HPCStatus_t integral<half_t, 1, half_t, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t integral<half_t, 3, half_t, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t integral<half_t, 4, half_t, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
#endif
 
#if defined(FASTCV_USE_OCL)
    template HPCStatus_t integral<uchar, 1, int, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<int, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t integral<uchar, 3, int, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<int, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t integral<uchar, 4, int, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<int, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    template HPCStatus_t integral<float, 1, float, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t integral<float, 3, float, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t integral<float, 4, float, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    template HPCStatus_t integral<cl_half, 1, cl_half, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t integral<cl_half, 3, cl_half, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t integral<cl_half, 4, cl_half, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
#endif

   // template HPCStatus_t integral<double, 1, double, 1, 1>(const Mat<double, 1>& src, Mat<double, 1> *dst);
   // template HPCStatus_t integral<double, 3, double, 3, 3>(const Mat<double, 3>& src, Mat<double, 3> *dst);
   // template HPCStatus_t integral<double, 4, double, 4, 4>(const Mat<double, 4>& src, Mat<double, 4> *dst);


}};
