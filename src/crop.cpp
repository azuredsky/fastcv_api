#include <status.h>
#include <image.h>
#include <fastcv.hpp>
namespace HPC {namespace fastcv {
    template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t crop(const Mat<Tsrc, ncSrc, type>& src, Point2i p, Tdst ratio, Mat<Tdst, ncDst, type> *dst) {
#if __cplusplus >= 201103L
            static_assert(nc<=ncSrc && nc<=ncDst, CHECK_CH_MACRO);
#endif

            if(src.numDimensions() == 0 || dst->numDimensions() == 0) return HPC_NOT_INITIALIZED;

            int sw = src.width();
            int sh = src.height();
            int sstep = src.widthStep()/sizeof(Tsrc);

           // const Tsrc* sp = src.ptr();
            int dw = dst->width();
            int dh = dst->height();
            int dstep = dst->widthStep()/sizeof(Tdst);
           // Tdst* dp = dst->ptr();
			void* sp = src.ptr();
			void* dp = dst->ptr();

            if(p.x < 0 || p.y < 0 || p.x > sw || p.y > sh) return HPC_INVALID_ARGS;
            if(p.x + dw > sw || p.y + dh > sh) return HPC_INVALID_ARGS;

            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                x86ImageCrop<Tsrc, ncSrc, Tdst, ncDst, nc>(
                    p.y, p.x, sstep, (const Tsrc*)sp,
                    dh, dw, dstep, (Tdst*)dp,
                    (float) ratio);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                armImageCrop<Tsrc, ncSrc, Tdst, ncDst, nc>(
                    p.y, p.x, sstep, (const Tsrc*)sp,
                    dh, dw, dstep, (Tdst*)dp,
                    (float) ratio);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                int device;
                cudaGetDevice(&device);
                cudaImageCrop<Tsrc, ncSrc, Tdst, ncDst, nc>(device, 0,
                    p.y, p.x, sstep, (const Tsrc*)sp,
                    dh, dw, dstep, (Tdst*)dp,
                    (float) ratio);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                oclImageCrop<Tsrc, ncSrc, Tdst, ncDst, nc>(
                    opencl.getCommandQueue(),
                    p.y, p.x, sstep, (cl_mem)sp,
                    dh, dw, dstep, (cl_mem)dp,
                    (float) ratio);
#endif
            }
            return HPC_SUCCESS;
        }

#if defined(FASTCV_USE_X86)
        template HPCStatus_t crop<uchar, 1, uchar, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, Point2i p, uchar ratio, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
        template HPCStatus_t crop<uchar, 3, uchar, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Point2i p, uchar ratio, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
        template HPCStatus_t crop<uchar, 4, uchar, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, Point2i p, uchar ratio, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst);
        template HPCStatus_t crop<float, 1, float, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& src, Point2i p, float ratio, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *dst);
        template HPCStatus_t crop<float, 3, float, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Point2i p, float ratio, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
        template HPCStatus_t crop<float, 4, float, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& src, Point2i p, float ratio, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *dst);
#endif

#if defined(FASTCV_USE_ARM)
        template HPCStatus_t crop<uchar, 1, uchar, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, Point2i p, uchar ratio, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
        template HPCStatus_t crop<uchar, 3, uchar, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Point2i p, uchar ratio, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
        template HPCStatus_t crop<uchar, 4, uchar, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, Point2i p, uchar ratio, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst);
        template HPCStatus_t crop<float, 1, float, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& src, Point2i p, float ratio, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
        template HPCStatus_t crop<float, 3, float, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, Point2i p, float ratio, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
        template HPCStatus_t crop<float, 4, float, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& src, Point2i p, float ratio, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *dst);
#endif

#if defined(FASTCV_USE_CUDA)
        template HPCStatus_t crop<uchar, 1, uchar, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Point2i p, uchar ratio, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t crop<uchar, 3, uchar, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Point2i p, uchar ratio, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t crop<uchar, 4, uchar, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Point2i p, uchar ratio, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t crop<float, 1, float, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Point2i p, float ratio, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t crop<float, 3, float, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Point2i p, float ratio, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t crop<float, 4, float, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Point2i p, float ratio, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t crop<half_t, 1, half_t, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Point2i p, half_t ratio, Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t crop<half_t, 3, half_t, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Point2i p, half_t ratio, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t crop<half_t, 4, half_t, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Point2i p, half_t ratio, Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
#endif

#if defined(FASTCV_USE_OCL)
        template HPCStatus_t crop<uchar, 1, uchar, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, Point2i p, uchar ratio, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t crop<uchar, 3, uchar, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Point2i p, uchar ratio, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t crop<uchar, 4, uchar, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, Point2i p, uchar ratio, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t crop<float, 1, float, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& src, Point2i p, float ratio, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t crop<float, 3, float, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Point2i p, float ratio, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t crop<float, 4, float, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& src, Point2i p, float ratio, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t crop<cl_half, 1, cl_half, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL>& src, Point2i p, cl_half ratio, Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t crop<cl_half, 3, cl_half, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, Point2i p, cl_half ratio, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t crop<cl_half, 4, cl_half, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL>& src, Point2i p, cl_half ratio, Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
#endif
}};
