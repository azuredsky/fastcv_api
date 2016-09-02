#include <image.h>
#include <fastcv.hpp>

namespace HPC { namespace fastcv {
    template<typename T, int ncSrc, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t remap(const Mat<T, ncSrc, type>& src, const Mat<float, 1, type>& mapX,const Mat<float,1, type>& mapY, InterpolationType mode, Mat<T, ncDst, type> *dst) {
#if __cplusplus >= 201103L
            static_assert(nc<=ncSrc && nc<=ncDst, CHECK_CH_MACRO);
#endif
            if(src.ptr() == NULL) return HPC_POINTER_NULL;
            if(mapX.ptr() == NULL) return HPC_POINTER_NULL;
            if(mapY.ptr() == NULL) return HPC_POINTER_NULL;
            if(dst->ptr() == NULL) return HPC_POINTER_NULL;
            
            if(NULL == dst) return HPC_INVALID_ARGS;
            if(src.numDimensions() == 0 || dst->numDimensions() == 0) return HPC_NOT_INITIALIZED;
            int sw = src.width();
            int sh = src.height();
            int sstep = src.widthStep()/sizeof(T);
            int dw = dst->width();
            int dh = dst->height();
            int dstep = dst->widthStep()/sizeof(T);

			void* sp = src.ptr();
			void* dp = dst->ptr();
			void* xp = mapX.ptr();
			void* yp = mapY.ptr();

            //if(sw != dw || sh != dh) return HPC_INVALID_ARGS;

            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                switch(mode) {
                case INTERPOLATION_TYPE_LINEAR:
                    x86RemapLinear<T, ncSrc, ncDst, nc>(
                        sh, sw, sstep, (T *)sp,
                        dh, dw, dstep, (T *)dp,
                        (float *)xp, (float *)yp);
                    break;
             /*   case INTERPOLATION_TYPE_NEAREST_POINT:
                    x86RemapNearestPoint<T, ncSrc, ncDst, nc>(
                        sh, sw, sstep, (T *)sp,
                        dh, dw, dstep, (T *)dp,
                        (float *)xp, (float *)yp);*/
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
                    armRemapLinear<T, ncSrc, ncDst, nc>(
                        sh, sw, sstep, (T *)sp,
                        dh, dw, dstep, (T *)dp,
                        (float *)xp, (float *)yp);
                    break;
                /*case INTERPOLATION_TYPE_NEAREST_POINT:
                    armRemapNearestPoint<T, ncSrc, ncDst, nc>(
                        sh, sw, sstep, (T *)sp,
                        dh, dw, dstep, (T *)dp,
                        (float *)xp, (float *)yp);
                    break;*/
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
                    cudaRemapLinear<T, ncSrc, ncDst, nc>(device, 0,
                        sh, sw, sstep, (T *)sp,
                        dh, dw, dstep, (T *)dp,
                        (float *)xp, (float *)yp);
                    break;
                case INTERPOLATION_TYPE_NEAREST_POINT:
                    cudaRemapNearestPoint<T, ncSrc, ncDst, nc>(device, 0,
                        sh, sw, sstep, (T *)sp,
                        dh, dw, dstep, (T *)dp,
                        (float *)xp, (float *)yp);
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
                    oclRemapLinear<T, ncSrc, ncDst, nc>(
                        opencl.getCommandQueue(),
                        sh, sw, sstep, (cl_mem)sp,
                        dh, dw, dstep, (cl_mem)dp,
                        (cl_mem)xp,(cl_mem)yp);
                    break;
                    /* case INTERPOLATION_TYPE_NEAREST_POINT:
                       oclWarpAffineNearestPoint<T, ncSrc, ncDst, nc>(
                       commandQueue,
                       sh, sw, sstep, (cl_mem)sp,
                       dh, dw, dsetp, (cl_mem)dp,
                       (cl_mem)xp, (cl_mem)yp);
                       break;*/
                default:
                    return HPC_NOT_IMPLEMENTED;
                }
#endif
            }
            return HPC_SUCCESS;
        }

#if defined(FASTCV_USE_X86)
        template HPCStatus_t remap<uchar, 1, 1, 1, EcoEnv_t::ECO_ENV_X86>(
              const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_X86>& mapY, InterpolationType mode, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
        template HPCStatus_t remap<uchar, 3, 3, 3, EcoEnv_t::ECO_ENV_X86>(
              const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_X86>& mapY, InterpolationType mode, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
        template HPCStatus_t remap<uchar, 4, 4, 4, EcoEnv_t::ECO_ENV_X86>(
              const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_X86>& mapY, InterpolationType mode, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst);

        template HPCStatus_t remap<float, 1, 1, 1, EcoEnv_t::ECO_ENV_X86>(
              const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_X86>& mapY, InterpolationType mode, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *dst);
        template HPCStatus_t remap<float, 3, 3, 3, EcoEnv_t::ECO_ENV_X86>(
              const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_X86>& mapY, InterpolationType mode, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
        template HPCStatus_t remap<float, 4, 4, 4, EcoEnv_t::ECO_ENV_X86>(
              const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_X86>& mapY, InterpolationType mode, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *dst);
#endif

#if defined(FASTCV_USE_ARM)
        template HPCStatus_t remap<uchar, 1, 1, 1, EcoEnv_t::ECO_ENV_ARM>(
              const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_ARM>& mapY, InterpolationType mode, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
        template HPCStatus_t remap<uchar, 3, 3, 3, EcoEnv_t::ECO_ENV_ARM>(
              const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_ARM>& mapY, InterpolationType mode, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
        template HPCStatus_t remap<uchar, 4, 4, 4, EcoEnv_t::ECO_ENV_ARM>(
              const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_ARM>& mapY, InterpolationType mode, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst);

/*        template HPCStatus_t remap<float, 1, 1, 1, EcoEnv_t::ECO_ENV_ARM>(
              const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_ARM>& mapY, InterpolationType mode, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
        template HPCStatus_t remap<float, 3, 3, 3, EcoEnv_t::ECO_ENV_ARM>(
              const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_ARM>& mapY, InterpolationType mode, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
        template HPCStatus_t remap<float, 4, 4, 4, EcoEnv_t::ECO_ENV_ARM>(
              const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_ARM>& mapY, InterpolationType mode, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *dst);
*/
#endif

#if defined(FASTCV_USE_CUDA)
        template HPCStatus_t remap<uchar, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(
              const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_CUDA>& mapY, InterpolationType mode, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t remap<uchar, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(
              const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_CUDA>& mapY, InterpolationType mode, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t remap<uchar, 4, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(
              const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_CUDA>& mapY, InterpolationType mode, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

        template HPCStatus_t remap<float, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(
              const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_CUDA>& mapY, InterpolationType mode, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t remap<float, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(
              const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_CUDA>& mapY, InterpolationType mode, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t remap<float, 4, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(
              const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_CUDA>& mapY, InterpolationType mode, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

              

        /*template HPCStatus_t remap<half_t, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(
              const Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA>& src, const Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA>& mapX,const Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA>& mapY, InterpolationType mode, Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t remap<half_t, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(
              const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, const Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA>& mapX,const Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA>& mapY, InterpolationType mode, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
        template HPCStatus_t remap<half_t, 4, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(
              const Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA>& src, const Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA>& mapX,const Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA>& mapY, InterpolationType mode, Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);*/
#endif

#if defined(FASTCV_USE_OCL)
        template HPCStatus_t remap<uchar, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(
              const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_OCL>& mapY, InterpolationType mode, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t remap<uchar, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(
              const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_OCL>& mapY, InterpolationType mode, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t remap<uchar, 4, 4, 4, EcoEnv_t::ECO_ENV_OCL>(
              const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_OCL>& mapY, InterpolationType mode, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

        template HPCStatus_t remap<float, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(
              const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_OCL>& mapY, InterpolationType mode, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t remap<float, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(
              const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_OCL>& mapY, InterpolationType mode, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t remap<float, 4, 4, 4, EcoEnv_t::ECO_ENV_OCL>(
              const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& mapX,const Mat<float,1, EcoEnv_t::ECO_ENV_OCL>& mapY, InterpolationType mode, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

        /*template HPCStatus_t remap<cl_half, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(
              const Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL>& src, const Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL>& mapX,const Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL>& mapY, InterpolationType mode, Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t remap<cl_half, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(
              const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, const Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL>& mapX,const Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL>& mapY, InterpolationType mode, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
        template HPCStatus_t remap<cl_half, 4, 4, 4, EcoEnv_t::ECO_ENV_OCL>(
              const Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL>& src, const Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL>& mapX,const Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL>& mapY, InterpolationType mode, Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL> *dst);*/
#endif
}}
