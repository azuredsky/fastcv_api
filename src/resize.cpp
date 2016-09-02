#include <image.h>
#include <fastcv.hpp>
namespace HPC { namespace fastcv {
    template<InterpolationType ip, typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc, EcoEnv_t type> HPCStatus_t resize(const Mat<Tsrc, ncSrc, type>& src,  Mat<Tdst, ncDst, type> *dst){
#if __cplusplus >= 201103L
        static_assert(nc<=ncSrc && nc<=ncDst, CHECK_CH_MACRO);
#endif
        if(src.ptr() == NULL) return HPC_POINTER_NULL;
        if(dst->ptr() == NULL) return HPC_POINTER_NULL;
        if(NULL == dst) return HPC_INVALID_ARGS;
        if(src.numDimensions() == 0 || dst->numDimensions() == 0) return HPC_NOT_INITIALIZED;
        int sh = src.height();
        int sw = src.width();
        int sstep = src.widthStep()/sizeof(Tsrc);
        //const Tsrc* sp = src.ptr();

        int dw = dst->width();
        int dh = dst->height();
        int dstep = dst->widthStep()/sizeof(Tdst);
        //Tdst* dp = dst->ptr();
        void* sp = src.ptr();
        void* dp = dst->ptr();
        if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
            switch(ip) {
            case INTERPOLATION_TYPE_LINEAR:
                x86ResizeLinear<Tsrc, ncSrc, Tdst, ncDst, nc>(sh, sw, sstep, (const Tsrc*)sp,
                    dh, dw, dstep, (Tdst*)dp);
                break;
            case INTERPOLATION_TYPE_NEAREST_POINT:
                x86ResizeNearestPoint<Tsrc, ncSrc, Tdst, ncDst, nc>(sh, sw,
                    sstep, (const Tsrc*)sp, dh, dw, dstep, (Tdst*)dp);
                break;
            case INTERPOLATION_TYPE_INTER_AREA:
                x86ResizeArea<Tsrc, ncSrc, Tdst, ncDst, nc>(sh, sw, 
                    sstep, (const Tsrc*)sp, dh, dw, dstep, (Tdst*)dp);
                break;
            default:
                return HPC_NOT_IMPLEMENTED;
            }
#endif
        }
        else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
            switch(ip) {
            case INTERPOLATION_TYPE_LINEAR:
                armResizeLinear<Tsrc, ncSrc, Tdst, ncDst, nc>(sh, sw, sstep, (const Tsrc*)sp,
                    dh, dw, dstep, (Tdst*)dp);
                break;
            case INTERPOLATION_TYPE_NEAREST_POINT:
                armResizeNearestPoint<Tsrc, ncSrc, Tdst, ncDst, nc>(sh, sw,
                    sstep, (const Tsrc*)sp, dh, dw, dstep, (Tdst*)dp);
                break;
            case INTERPOLATION_TYPE_INTER_AREA:
                armResizeArea<Tsrc, ncSrc, Tdst, ncDst, nc>(sh, sw, 
                    sstep, (const Tsrc*)sp, dh, dw, dstep, (Tdst*)dp);
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
            switch(ip) {
            case INTERPOLATION_TYPE_LINEAR:
                cudaResizeLinear<Tsrc, ncSrc, Tdst, ncDst, nc>(device, 0,
                    sh, sw, sstep, (const Tsrc*)sp, dh, dw, dstep, (Tdst*)dp);
                break;
            case INTERPOLATION_TYPE_NEAREST_POINT:
                cudaResizeNearestPoint<Tsrc, ncSrc, Tdst, ncDst, nc>(device, 0,
                    sh, sw, sstep, (const Tsrc*)sp, dh, dw, dstep, (Tdst*)dp);
                break;
            default:
                return HPC_NOT_IMPLEMENTED;
            }
#endif
        }
        else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
            switch(ip) {
            case INTERPOLATION_TYPE_LINEAR:
                oclResizeLinear<Tsrc, ncSrc, Tdst, ncDst, nc>(
                    opencl.getCommandQueue(),
                    sh, sw, sstep, (cl_mem)sp,
                    dh, dw, dstep, (cl_mem)dp);
                break;
            case INTERPOLATION_TYPE_NEAREST_POINT:
                oclResizeNearestPoint<Tsrc, ncSrc, Tdst, ncDst, nc>(
                    opencl.getCommandQueue(),
                    sh, sw, sstep, (cl_mem)sp,
                    dh, dw, dstep, (cl_mem)dp);
                break;
            default:
                return HPC_NOT_IMPLEMENTED;
            }
#endif
        }
        return HPC_SUCCESS;
    }

#if defined(FASTCV_USE_X86)
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, uchar, 1, uchar, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, uchar, 3, uchar, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, uchar, 4, uchar, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, uchar, 1, uchar, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, uchar, 3, uchar, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, uchar, 4, uchar, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, uchar, 1, uchar, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, uchar, 3, uchar, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, uchar, 4, uchar, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst);


    /*
    template HPCStatus_t resize<uchar, 1, uchar, 1, 1, INTERPOLATION_TYPE_INTER_AREA>(const Mat<uchar, 1>& src, Mat<uchar, 1> *dst);
    template HPCStatus_t resize<uchar, 3, uchar, 3, 3, INTERPOLATION_TYPE_INTER_AREA>(const Mat<uchar, 3>& src, Mat<uchar, 3> *dst);
    template HPCStatus_t resize<uchar, 4, uchar, 4, 4, INTERPOLATION_TYPE_INTER_AREA>(const Mat<uchar, 4>& src,  Mat<uchar, 4> *dst);
    */

    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, float, 1, float, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, float, 3, float, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, float, 4, float, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, float, 1, float, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, float, 3, float, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, float, 4, float, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, float, 1, float, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, float, 3, float, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, float, 4, float, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *dst);

/*
    template HPCStatus_t resize<float, 1, float, 1, 1, INTERPOLATION_TYPE_INTER_AREA>(const Mat<float, 1>& src, Mat<float, 1> *dst);
    template HPCStatus_t resize<float, 3, float, 3, 3, INTERPOLATION_TYPE_INTER_AREA>(const Mat<float, 3>& src, Mat<float, 3> *dst);
    template HPCStatus_t resize<float, 4, float, 4, 4, INTERPOLATION_TYPE_INTER_AREA>(const Mat<float, 4>& src,  Mat<float, 4> *dst);
*/
#endif

#if defined(FASTCV_USE_ARM)
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, uchar, 1, uchar, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, uchar, 3, uchar, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, uchar, 4, uchar, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, uchar, 1, uchar, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, uchar, 3, uchar, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, uchar, 4, uchar, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, uchar, 1, uchar, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, uchar, 3, uchar, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, uchar, 4, uchar, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst);


    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, float, 1, float, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, float, 3, float, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, float, 4, float, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, float, 1, float, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, float, 3, float, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, float, 4, float, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, float, 1, float, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, float, 3, float, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, float, 4, float, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *dst);

#endif

#if defined(FASTCV_USE_CUDA)
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, uchar, 1, uchar, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, uchar, 3, uchar, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, uchar, 4, uchar, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, uchar, 1, uchar, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, uchar, 3, uchar, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, uchar, 4, uchar, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, uchar, 1, uchar, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, uchar, 3, uchar, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, uchar, 4, uchar, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, float, 1, float, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, float, 3, float, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, float, 4, float, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, float, 1, float, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, float, 3, float, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, float, 4, float, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, float, 1, float, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, float, 3, float, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, float, 4, float, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, half_t, 1, half_t, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, half_t, 3, half_t, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, half_t, 4, half_t, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, half_t, 1, half_t, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, half_t, 3, half_t, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, half_t, 4, half_t, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA>& src,  Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, half_t, 1, half_t, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, half_t, 3, half_t, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, half_t, 4, half_t, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

#endif

#if defined(FASTCV_USE_OCL)
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, uchar, 1, uchar, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, uchar, 3, uchar, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, uchar, 4, uchar, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, uchar, 1, uchar, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, uchar, 3, uchar, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, uchar, 4, uchar, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, uchar, 1, uchar, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, uchar, 3, uchar, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, uchar, 4, uchar, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, float, 1, float, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, float, 3, float, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, float, 4, float, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, float, 1, float, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, float, 3, float, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, float, 4, float, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, float, 1, float, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, float, 3, float, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, float, 4, float, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, cl_half, 1, cl_half, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, cl_half, 3, cl_half, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_NEAREST_POINT, cl_half, 4, cl_half, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, cl_half, 1, cl_half, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, cl_half, 3, cl_half, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_LINEAR, cl_half, 4, cl_half, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL>& src,  Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, cl_half, 1, cl_half, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, cl_half, 3, cl_half, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t resize<INTERPOLATION_TYPE_INTER_AREA, cl_half, 4, cl_half, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
#endif
}}
