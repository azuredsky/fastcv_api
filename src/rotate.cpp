//#include <status.h>
#include <image.h>
#include <fastcv.hpp>

namespace HPC { namespace fastcv {
    template<typename T, int ncSrc, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t rotate(const Mat<T, ncSrc, type>& src, Mat<T, ncDst,type>* dst, float degree) {
            if(src.ptr() == NULL) return HPC_POINTER_NULL;
            if(dst->ptr() == NULL) return HPC_POINTER_NULL;
            if(src.numDimensions() == 0 || dst->numDimensions() == 0) return HPC_NOT_INITIALIZED;
            if(fabs(degree - 90) > 1e-9 && fabs(degree - 180) > 1e-9 && fabs(degree - 270) > 1e-9)
                return HPC_INVALID_ARGS;

            int sw = src.width();
            int sh = src.height();
            int sstep = src.widthStep()/sizeof(T);
            int dw = dst->width();
            int dh = dst->height();
            int dstep = dst->widthStep()/sizeof(T);
            if(fabs(degree - 90) < 1e-9 && (sh != dw || sw != dh))
                return HPC_INVALID_ARGS;
            if(fabs(degree - 180) < 1e-9 && (sh != dh || sw != dw))
                return HPC_INVALID_ARGS;
            if(fabs(degree - 270) < 1e-9 && (sh != dw || sw != dh))
                return HPC_INVALID_ARGS;
            int deg = 0;
            if(fabs(degree - 90) < 1e-9)
                deg = 90;
            else if (fabs(degree - 180) < 1e-9)
                deg = 180;
            else if (fabs(degree - 270) < 1e-9)
                deg = 270;
            else
                return HPC_INVALID_ARGS;

            void* sp = src.ptr();
            void* dp = dst->ptr();

            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                x86RotateNx90degree<T, ncSrc, ncDst, nc>(sh, sw, sstep, (const T*)sp,
                    dh, dw, dstep, (T*)dp, deg);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                armRotateNx90degree<T, ncSrc, ncDst, nc>(sh, sw, sstep, (const T*)sp,
                    dh, dw, dstep, (T*)dp, deg);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                int device;
                cudaGetDevice(&device);
                cudaRotateNx90degree<T, ncSrc, ncDst, nc>(device, 0,
                    sh, sw, sstep, (const T*)sp, dh, dw, dstep, (T*)dp, deg);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
//                oclRotateNx90degree<T, ncSrc, ncDst, nc>(opencl.getCommandQueue(),
//                    sh, sw, sstep, (cl_mem)sp, dh, dw, dstep, (cl_mem)dp, deg);

#endif
            }
            return HPC_SUCCESS;
        }

    template<YUV420Type yt, typename T, EcoEnv_t type>
        HPCStatus_t rotate_YUV420(const Mat<T, 1, type>& src, Mat<T, 1, type>* dst, float degree) {
            if(NULL == dst) return HPC_POINTER_NULL;
            if(src.numDimensions() == 0 || dst->numDimensions() == 0) return HPC_NOT_INITIALIZED;
            if(fabs(degree - 90) > 1e-9 && fabs(degree - 180) > 1e-9 && fabs(degree - 270) > 1e-9)
                return HPC_INVALID_ARGS;

            int sw = src.width();
            int sh = src.height();
            int sstep = src.widthStep()/sizeof(T);
            int dw = dst->width();
            int dh = dst->height();
            int dstep = dst->widthStep()/sizeof(T);

            int Ysh = sh*2/3;
            int Ysw = sw;
            int Ydh = dh*2/3;
            int Ydw = dw;

            if(fabs(degree - 90) < 1e-9 && (Ysh != Ydw || Ysw != Ydh))
                return HPC_INVALID_ARGS;
            if(fabs(degree - 180) < 1e-9 && (Ysh != Ydh || Ysw != Ydw))
                return HPC_INVALID_ARGS;
            if(fabs(degree - 270) < 1e-9 && (Ysh != Ydw || Ysw != Ydh))
                return HPC_INVALID_ARGS;
            int deg = 0;
            if(fabs(degree - 90) < 1e-9)
                deg = 90;
            else if (fabs(degree - 180) < 1e-9)
                deg = 180;
            else if (fabs(degree - 270) < 1e-9)
                deg = 270;
            else
                return HPC_INVALID_ARGS;

            void* sp = src.ptr();
            void* dp = dst->ptr();

            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                switch(yt) {
                case YUV420_NV12:
                case YUV420_NV21:
                    x86RotateNx90degree_YUV420<T>(sh, sw, sstep, (const T*)sp,
                        dh, dw, dstep, (T*)dp, deg, 0);
                    break;
                case YUV420_I420:
                    x86RotateNx90degree_YUV420<T>(sh, sw, sstep, (const T*)sp,
                        dh, dw, dstep, (T*)dp, deg, 1);
                    break;
                default:
                    return HPC_NOT_IMPLEMENTED;
                }
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                switch(yt) {
                case YUV420_NV12:
                case YUV420_NV21:
                    armRotateNx90degree_YUV420<T>(sh, sw, sstep, (const T*)sp,
                        dh, dw, dstep, (T*)dp, deg, 0);
                    break;
                case YUV420_I420:
                    armRotateNx90degree_YUV420<T>(sh, sw, sstep, (const T*)sp,
                        dh, dw, dstep, (T*)dp, deg, 1);
                    break;

                default:
                    return HPC_NOT_IMPLEMENTED;
                }
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)

				int device;


                cudaGetDevice(&device);
                switch(yt) {
                case YUV420_NV12:
                case YUV420_NV21:
                    cudaRotateNx90degree_YUV420<T>(device,0,sh, sw, sstep, (const T*)sp,
                        dh, dw, dstep, (T*)dp, deg, 0);
                    break;
                case YUV420_I420:
                    cudaRotateNx90degree_YUV420<T>(device,0,sh, sw, sstep, (const T*)sp,
                        dh, dw, dstep, (T*)dp, deg, 1);
                    break;
                default:
                    return HPC_NOT_IMPLEMENTED;
                }
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
#endif
            }
            return HPC_SUCCESS;
        }

#if defined(FASTCV_USE_X86)
    template HPCStatus_t rotate<float, 1, 1, 1,EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1,EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 1,EcoEnv_t::ECO_ENV_X86> *dst, float degree);
    template HPCStatus_t rotate<float, 3, 3, 3,EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3,EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 3,EcoEnv_t::ECO_ENV_X86> *dst, float degree);
    template HPCStatus_t rotate<float, 4, 4, 4,EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4,EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 4,EcoEnv_t::ECO_ENV_X86> *dst, float degree);

    template HPCStatus_t rotate<uchar, 1, 1, 1,EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1,EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1,EcoEnv_t::ECO_ENV_X86> *dst, float degree);
    template HPCStatus_t rotate<uchar, 3, 3, 3,EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3,EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3,EcoEnv_t::ECO_ENV_X86> *dst, float degree);
    template HPCStatus_t rotate<uchar, 4, 4, 4,EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4,EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 4,EcoEnv_t::ECO_ENV_X86> *dst, float degree);

    template HPCStatus_t rotate_YUV420<YUV420_NV12, uchar,EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1,EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1,EcoEnv_t::ECO_ENV_X86> *dst, float degree);
    template HPCStatus_t rotate_YUV420<YUV420_NV21, uchar,EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1,EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1,EcoEnv_t::ECO_ENV_X86> *dst, float degree);
    template HPCStatus_t rotate_YUV420<YUV420_I420, uchar,EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1,EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1,EcoEnv_t::ECO_ENV_X86> *dst, float degree);

#endif

#if defined(FASTCV_USE_ARM)
    template HPCStatus_t rotate<float, 1, 1, 1,EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1,EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 1,EcoEnv_t::ECO_ENV_ARM> *dst, float degree);
    template HPCStatus_t rotate<float, 3, 3, 3,EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3,EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 3,EcoEnv_t::ECO_ENV_ARM> *dst, float degree);
    template HPCStatus_t rotate<float, 4, 4, 4,EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4,EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 4,EcoEnv_t::ECO_ENV_ARM> *dst, float degree);

    template HPCStatus_t rotate<uchar, 1, 1, 1,EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1,EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1,EcoEnv_t::ECO_ENV_ARM> *dst, float degree);
    template HPCStatus_t rotate<uchar, 3, 3, 3,EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3,EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3,EcoEnv_t::ECO_ENV_ARM> *dst, float degree);
    template HPCStatus_t rotate<uchar, 4, 4, 4,EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4,EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 4,EcoEnv_t::ECO_ENV_ARM> *dst, float degree);

    template HPCStatus_t rotate_YUV420<YUV420_NV12, uchar,EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1,EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1,EcoEnv_t::ECO_ENV_ARM> *dst, float degree);
    template HPCStatus_t rotate_YUV420<YUV420_NV21, uchar,EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1,EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1,EcoEnv_t::ECO_ENV_ARM> *dst, float degree);
    template HPCStatus_t rotate_YUV420<YUV420_I420, uchar,EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1,EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1,EcoEnv_t::ECO_ENV_ARM> *dst, float degree);

#endif

#if defined(FASTCV_USE_CUDA)
    template HPCStatus_t rotate<float, 1, 1, 1,EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1,EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 1,EcoEnv_t::ECO_ENV_CUDA> *dst, float degree);
    template HPCStatus_t rotate<float, 3, 3, 3,EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3,EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 3,EcoEnv_t::ECO_ENV_CUDA> *dst, float degree);
    template HPCStatus_t rotate<float, 4, 4, 4,EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4,EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 4,EcoEnv_t::ECO_ENV_CUDA> *dst, float degree);

    template HPCStatus_t rotate<uchar, 1, 1, 1,EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1,EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1,EcoEnv_t::ECO_ENV_CUDA> *dst, float degree);
    template HPCStatus_t rotate<uchar, 3, 3, 3,EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3,EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3,EcoEnv_t::ECO_ENV_CUDA> *dst, float degree);
    template HPCStatus_t rotate<uchar, 4, 4, 4,EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4,EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 4,EcoEnv_t::ECO_ENV_CUDA> *dst, float degree);

    template HPCStatus_t rotate_YUV420<YUV420_NV12, uchar,EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1,EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1,EcoEnv_t::ECO_ENV_CUDA> *dst, float degree);
    template HPCStatus_t rotate_YUV420<YUV420_NV21, uchar,EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1,EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1,EcoEnv_t::ECO_ENV_CUDA> *dst, float degree);
    template HPCStatus_t rotate_YUV420<YUV420_I420, uchar,EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1,EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1,EcoEnv_t::ECO_ENV_CUDA> *dst, float degree);
#endif

#if defined(FASTCV_USE_OCL)
    template HPCStatus_t rotate<float, 1, 1, 1,EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1,EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 1,EcoEnv_t::ECO_ENV_OCL> *dst, float degree);
    template HPCStatus_t rotate<float, 3, 3, 3,EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3,EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 3,EcoEnv_t::ECO_ENV_OCL> *dst, float degree);
    template HPCStatus_t rotate<float, 4, 4, 4,EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4,EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 4,EcoEnv_t::ECO_ENV_OCL> *dst, float degree);

    template HPCStatus_t rotate<uchar, 1, 1, 1,EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1,EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1,EcoEnv_t::ECO_ENV_OCL> *dst, float degree);
    template HPCStatus_t rotate<uchar, 3, 3, 3,EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3,EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3,EcoEnv_t::ECO_ENV_OCL> *dst, float degree);
    template HPCStatus_t rotate<uchar, 4, 4, 4,EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4,EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 4,EcoEnv_t::ECO_ENV_OCL> *dst, float degree);

    template HPCStatus_t rotate_YUV420<YUV420_NV12, uchar,EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1,EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1,EcoEnv_t::ECO_ENV_OCL> *dst, float degree);
    template HPCStatus_t rotate_YUV420<YUV420_NV21, uchar,EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1,EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1,EcoEnv_t::ECO_ENV_OCL> *dst, float degree);
    template HPCStatus_t rotate_YUV420<YUV420_I420, uchar,EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1,EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1,EcoEnv_t::ECO_ENV_OCL> *dst, float degree);
#endif
}};
