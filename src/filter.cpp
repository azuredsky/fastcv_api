#include <fastcv.hpp>
#include <image.h>

namespace HPC { namespace fastcv {
    template<BorderType bt, typename T, int filterSize, int ncSrc, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t filter2D(const Mat<T, ncSrc, type>& src, Mat<T, ncDst, type> *dst, const MatX2D<float, filterSize, filterSize>& f) {
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
                if(bt == BORDER_DEFAULT) {
                    x86Filter2DReflect101<T, ncSrc, ncDst, nc>(sh, sw, sstep, (const T*)sp,
                        filterSize, f.ptr(), dstep, (T*)dp);
                } else {
                    return HPC_NOT_SUPPORTED;
                }
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                if(bt == BORDER_DEFAULT) {
                    armFilter2DReflect101<T, ncSrc, ncDst, nc>(sh, sw, sstep, (const T*)sp,
                        filterSize, f.ptr(), dstep, (T*)dp);
                } else {
                    return HPC_NOT_SUPPORTED;
                }
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                if(bt == BORDER_DEFAULT) {
                    cudaFilter2DReflect101<T, ncSrc, ncDst, nc>(0,0,sh, sw, sstep,
                        (const T*)sp, filterSize, f.ptr(), dstep, (T*)dp);
                } else {
                    return HPC_NOT_SUPPORTED;
                }
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                if(bt == BORDER_DEFAULT) {
                    oclFilter2DReflect101<T,ncSrc,ncDst,nc>(opencl.getCommandQueue(),sh,sw,sstep,(cl_mem)sp,filterSize,f.ptr(),dstep,(cl_mem)dp);
                }else {
                    return HPC_NOT_SUPPORTED;
                }
#endif
            }
            return HPC_SUCCESS;
        }
#if defined(FASTCV_USE_X86)
    template HPCStatus_t filter2D<BORDER_DEFAULT, uchar, 3, 1, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst, const MatX2D<float, 3, 3>& f);
    template HPCStatus_t filter2D<BORDER_DEFAULT, uchar, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst, const MatX2D<float, 3, 3>& f);
    template HPCStatus_t filter2D<BORDER_DEFAULT, uchar, 3, 4, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst, const MatX2D<float, 3, 3>& f);

    template HPCStatus_t filter2D<BORDER_DEFAULT, float, 3, 1, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *dst, const MatX2D<float, 3, 3>& f);
    template HPCStatus_t filter2D<BORDER_DEFAULT, float, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst, const MatX2D<float, 3, 3>& f);
    template HPCStatus_t filter2D<BORDER_DEFAULT, float, 3, 4, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *dst, const MatX2D<float, 3, 3>& f);
#endif

#if defined(FASTCV_USE_ARM)
    template HPCStatus_t filter2D<BORDER_DEFAULT, uchar, 3, 1, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst, const MatX2D<float, 3, 3>& f);
    template HPCStatus_t filter2D<BORDER_DEFAULT, uchar, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst, const MatX2D<float, 3, 3>& f);
    template HPCStatus_t filter2D<BORDER_DEFAULT, uchar, 3, 4, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst, const MatX2D<float, 3, 3>& f);

    template HPCStatus_t filter2D<BORDER_DEFAULT, float, 3, 1, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *dst, const MatX2D<float, 3, 3>& f);
    template HPCStatus_t filter2D<BORDER_DEFAULT, float, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst, const MatX2D<float, 3, 3>& f);
    template HPCStatus_t filter2D<BORDER_DEFAULT, float, 3, 4, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *dst, const MatX2D<float, 3, 3>& f);
#endif

#if defined(FASTCV_USE_CUDA)
    template HPCStatus_t filter2D<BORDER_DEFAULT, uchar, 3, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst, const MatX2D<float, 3, 3>& f);
    template HPCStatus_t filter2D<BORDER_DEFAULT, uchar, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst, const MatX2D<float, 3, 3>& f);
    template HPCStatus_t filter2D<BORDER_DEFAULT, uchar, 3, 4, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst, const MatX2D<float, 3, 3>& f);

    template HPCStatus_t filter2D<BORDER_DEFAULT, float, 3, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *dst, const MatX2D<float, 3, 3>& f);
    template HPCStatus_t filter2D<BORDER_DEFAULT, float, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst, const MatX2D<float, 3, 3>& f);
    template HPCStatus_t filter2D<BORDER_DEFAULT, float, 3, 4, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *dst, const MatX2D<float, 3, 3>& f);

    template HPCStatus_t filter2D<BORDER_DEFAULT, half_t, 3, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA> *dst, const MatX2D<float, 3, 3>& f);
    template HPCStatus_t filter2D<BORDER_DEFAULT, half_t, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst, const MatX2D<float, 3, 3>& f);
    template HPCStatus_t filter2D<BORDER_DEFAULT, half_t, 3, 4, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA> *dst, const MatX2D<float, 3, 3>& f);
#endif

#if defined(FASTCV_USE_OCL)
    template HPCStatus_t filter2D<BORDER_DEFAULT, uchar, 3, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst, const MatX2D<float, 3, 3>& f);
    template HPCStatus_t filter2D<BORDER_DEFAULT, uchar, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst, const MatX2D<float, 3, 3>& f);
    template HPCStatus_t filter2D<BORDER_DEFAULT, uchar, 3, 4, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst, const MatX2D<float, 3, 3>& f);

    template HPCStatus_t filter2D<BORDER_DEFAULT, float, 3, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *dst, const MatX2D<float, 3, 3>& f);
    template HPCStatus_t filter2D<BORDER_DEFAULT, float, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst, const MatX2D<float, 3, 3>& f);
    template HPCStatus_t filter2D<BORDER_DEFAULT, float, 3, 4, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *dst, const MatX2D<float, 3, 3>& f);

    template HPCStatus_t filter2D<BORDER_DEFAULT, cl_half, 3, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL> *dst, const MatX2D<float, 3, 3>& f);
    template HPCStatus_t filter2D<BORDER_DEFAULT, cl_half, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst, const MatX2D<float, 3, 3>& f);
    template HPCStatus_t filter2D<BORDER_DEFAULT, cl_half, 3, 4, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL> *dst, const MatX2D<float, 3, 3>& f);
#endif

  //  template HPCStatus_t filter2D<float, 5, 1, 1, 1, BORDER_DEFAULT>(const Mat<float, 1>& src, Mat<float, 1> *dst, const MatX2D<float, 5, 5>& f);
  //  template HPCStatus_t filter2D<float, 7, 1, 1, 1, BORDER_DEFAULT>(const Mat<float, 1>& src, Mat<float, 1> *dst, const MatX2D<float, 7, 7>& f);
}}
