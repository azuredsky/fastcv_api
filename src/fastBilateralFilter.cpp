#include <fastcv.hpp>
#include <image.h>

namespace HPC { namespace fastcv {

    template<BorderType bt, typename T, int ncSrc, int ncBase, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t fastBilateralFilter(const Mat<T, ncSrc, type>& src, const Mat<T, ncBase, type>& base, double color, double space, Mat<T, ncDst, type> *dst) {
#if __cplusplus >= 201103L
#endif

            if(src.ptr() == NULL) return HPC_POINTER_NULL;
            if(dst->ptr() == NULL) return HPC_POINTER_NULL;
            if(src.width() != base.width() || src.height() != base.height()) return HPC_INVALID_ARGS;
            if(src.width() != dst->width() || src.height() != dst->height()) return HPC_INVALID_ARGS;

            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                if(bt == BORDER_DEFAULT) {
                    armFastBilateralFilter<T, ncSrc, ncBase, ncDst, nc>(src.height(), src.width(),
                        src.width(), (T*)src.ptr(), base.width(), (T*)base.ptr(), color, space,
                        dst->width(), (T*)dst->ptr());
                } else {
                    return HPC_NOT_SUPPORTED;
                }
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
#endif
            }
            return HPC_SUCCESS;
        }

#if defined(FASTCV_USE_X86)
    template HPCStatus_t fastBilateralFilter<BORDER_DEFAULT, float, 1, 1, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& base, double color, double space, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t fastBilateralFilter<BORDER_DEFAULT, float, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& base, double color, double space, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t fastBilateralFilter<BORDER_DEFAULT, float, 4, 4, 4, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& src, const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& base, double color, double space, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *dst);
#endif

#if defined(FASTCV_USE_ARM)
    template HPCStatus_t fastBilateralFilter<BORDER_DEFAULT, float, 1, 1, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& base, double color, double space, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t fastBilateralFilter<BORDER_DEFAULT, float, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& base, double color, double space, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t fastBilateralFilter<BORDER_DEFAULT, float, 4, 4, 4, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& src, const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& base, double color, double space, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *dst);
#endif

#if defined(FASTCV_USE_CUDA)
    template HPCStatus_t fastBilateralFilter<BORDER_DEFAULT, float, 1, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& base, double color, double space, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t fastBilateralFilter<BORDER_DEFAULT, float, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& base, double color, double space, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t fastBilateralFilter<BORDER_DEFAULT, float, 4, 4, 4, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& src, const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& base, double color, double space, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
#endif

#if defined(FASTCV_USE_OCL)
    template HPCStatus_t fastBilateralFilter<BORDER_DEFAULT, float, 1, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& src, const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& base, double color, double space, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t fastBilateralFilter<BORDER_DEFAULT, float, 3, 3, 3, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& base, double color, double space, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t fastBilateralFilter<BORDER_DEFAULT, float, 4, 4, 4, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& src, const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& base, double color, double space, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
#endif
}}
