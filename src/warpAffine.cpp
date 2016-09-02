#include <image.h>
#include <fastcv.hpp>

namespace HPC { namespace fastcv {
    template<typename T, int ncSrc, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t warpAffine(const Mat<T, ncSrc, type>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<T, ncDst, type> *dst) {
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
        //    const T* sp = src.ptr();
            int dw = dst->width();
            int dh = dst->height();
             int dstep = dst->widthStep()/sizeof(T);
          // T* dp = dst->ptr();
			void* sp = src.ptr();
			void* dp = dst->ptr();

            //if(sw != dw || sh != dh) return HPC_INVALID_ARGS;

            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                switch(mode) {
                case INTERPOLATION_TYPE_LINEAR:
                    x86WarpAffineLinear<T, ncSrc, ncDst, nc>(
                        sh, sw, sstep, (const T*)sp,
                        dh, dw, dstep, (T*)dp,
                        map.ptr());
                    break;
                case INTERPOLATION_TYPE_NEAREST_POINT:
                    x86WarpAffineNearestPoint<T, ncSrc, ncDst, nc>(
                        sh, sw, sstep, (const T*)sp,
                        dh, dw, dstep, (T*)dp,
                        map.ptr());
                    break;
                default:
                    return HPC_NOT_IMPLEMENTED;
                }
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                switch(mode) {
                case INTERPOLATION_TYPE_LINEAR:
                    armWarpAffineLinear<T, ncSrc, ncDst, nc>(
                        sh, sw, sstep, (const T*)sp,
                        dh, dw, dstep, (T*)dp,
                        map.ptr());
                    break;
                case INTERPOLATION_TYPE_NEAREST_POINT:
                    armWarpAffineNearestPoint<T, ncSrc, ncDst, nc>(
                        sh, sw, sstep, (const T*)sp,
                        dh, dw, dstep, (T*)dp,
                        map.ptr());
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
                switch(mode) {
                case INTERPOLATION_TYPE_LINEAR:
                    //map11 = (float*)map.ptr();
                    cudaWarpAffineLinear<T, ncSrc, ncDst, nc>(device, 0,
                        sh, sw, sstep, (const T*)sp,
                        dh, dw, dstep, (T*)dp,
                        map.ptr());
                    break;
                case INTERPOLATION_TYPE_NEAREST_POINT:
                    cudaWarpAffineNearestPoint<T, ncSrc, ncDst, nc>(device, 0,
                        sh, sw, sstep, (const T*)sp,
                        dh, dw, dstep, (T*)dp,
                        map.ptr());
                    break;
                default:
                    return HPC_NOT_IMPLEMENTED;
                }
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                switch(mode) {
                case INTERPOLATION_TYPE_LINEAR:
                    oclWarpAffineLinear<T, ncSrc, ncDst, nc>(
                        opencl.getCommandQueue(),
                        sh, sw, sstep, (cl_mem)sp,
                        dh, dw, dstep, (cl_mem)dp,
                        map.ptr());
                    break;
                case INTERPOLATION_TYPE_NEAREST_POINT:
                    oclWarpAffineNearestPoint<T, ncSrc, ncDst, nc>(
                        opencl.getCommandQueue(),
                        sh, sw, sstep, (cl_mem)sp,
                        dh, dw, dstep, (cl_mem)dp,
                        map.ptr());
                    break;
                default:
                    return HPC_NOT_IMPLEMENTED;
                }
#endif
            }
            return HPC_SUCCESS;
        }

#if defined(FASTCV_USE_X86)
        template HPCStatus_t warpAffine<uchar, 1, 1, 1, EcoEnv_t::ECO_ENV_X86>(
              const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
        template HPCStatus_t warpAffine<uchar, 3, 3, 3, EcoEnv_t::ECO_ENV_X86>(
              const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
        template HPCStatus_t warpAffine<uchar, 4, 4, 4, EcoEnv_t::ECO_ENV_X86>(
              const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst);

        template HPCStatus_t warpAffine<float, 1, 1, 1, EcoEnv_t::ECO_ENV_X86>(
              const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *dst);
        template HPCStatus_t warpAffine<float, 3, 3, 3, EcoEnv_t::ECO_ENV_X86>(
              const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
        template HPCStatus_t warpAffine<float, 4, 4, 4, EcoEnv_t::ECO_ENV_X86>(
              const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *dst);
#endif

#if defined(FASTCV_USE_ARM)
        template HPCStatus_t warpAffine<uchar, 1, 1, 1, EcoEnv_t::ECO_ENV_ARM>(
              const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
        template HPCStatus_t warpAffine<uchar, 3, 3, 3, EcoEnv_t::ECO_ENV_ARM>(
              const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
        template HPCStatus_t warpAffine<uchar, 4, 4, 4, EcoEnv_t::ECO_ENV_ARM>(
              const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst);

        template HPCStatus_t warpAffine<float, 1, 1, 1, EcoEnv_t::ECO_ENV_ARM>(
              const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
        template HPCStatus_t warpAffine<float, 3, 3, 3, EcoEnv_t::ECO_ENV_ARM>(
              const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
        template HPCStatus_t warpAffine<float, 4, 4, 4, EcoEnv_t::ECO_ENV_ARM>(
              const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *dst);
#endif

#if defined(FASTCV_USE_CUDA)
        template HPCStatus_t warpAffine<uchar, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(
              const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t warpAffine<uchar, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(
              const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t warpAffine<uchar, 4, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(
              const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

        template HPCStatus_t warpAffine<float, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(
              const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t warpAffine<float, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(
              const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t warpAffine<float, 4, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(
              const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

        template HPCStatus_t warpAffine<half_t, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(
              const Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t warpAffine<half_t, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(
              const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t warpAffine<half_t, 4, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(
              const Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
#endif

#if defined(FASTCV_USE_OCL)
        template HPCStatus_t warpAffine<uchar, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(
              const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t warpAffine<uchar, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(
              const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t warpAffine<uchar, 4, 4, 4, EcoEnv_t::ECO_ENV_OCL>(
              const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

        template HPCStatus_t warpAffine<float, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(
              const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t warpAffine<float, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(
              const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t warpAffine<float, 4, 4, 4, EcoEnv_t::ECO_ENV_OCL>(
              const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

        template HPCStatus_t warpAffine<cl_half, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(
              const Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t warpAffine<cl_half, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(
              const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t warpAffine<cl_half, 4, 4, 4, EcoEnv_t::ECO_ENV_OCL>(
              const Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL>& src, const MatX2D<float, 3, 3>& map, InterpolationType mode, Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
#endif
}}
