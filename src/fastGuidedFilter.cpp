#include <fastcv.hpp>
#include <image.h>
#include <stdio.h>
#include "func_inline.hpp"

namespace HPC { namespace fastcv {
/*
    typedef enum {
        BORDER_CONSTANT = 0, //iiiiii|abcdefgh|iiiiiii with some specified i
        BORDER_REPLICATE = 1, // aaaaaa|abcdefgh|hhhhhhh
        BORDER_REFLECT = 2, // fedcba|abcdefgh|hgfedcb
        BORDER_WRAP    = 3, // cdefgh|abcdefgh|abcdefg
        BORDER_REFLECT101 = 4, // gfedcb|abcdefgh|gfedcba
        BORDER_DEFAULT = 4, // same as BORDER_REFLECT_101
        BORDER_TRANSPARENT = 5, // uvwxyz|absdefgh|i
        BORDER_ISOLATED = 6, //do not look outside of ROI
    } BorderType;
*/

    template<BorderType bt, typename T, int ncGuided, int ncSrc, int ncDst, int nc, EcoEnv_t type>
        HPCStatus_t fastGuidedFilter(const Mat<T, ncGuided, type>& guided, const Mat<T, ncSrc, type>& src, Mat<T, ncDst, type> *dst, int radius, float eps, int scale)
        {

#if __cplusplus >= 201103L
#endif
            if(guided.ptr() == NULL) return HPC_POINTER_NULL;
            if(src.ptr() == NULL) return HPC_POINTER_NULL;
            if(dst->ptr() == NULL) return HPC_POINTER_NULL;
            if(src.numDimensions() == 0 || dst->numDimensions() == 0) return HPC_NOT_INITIALIZED;
            
            int r_boxFilter = radius / scale;
            int ksize_boxFilter = 2 * r_boxFilter + 1;
            if(ksize_boxFilter != 3) {
                return HPC_NOT_SUPPORTED;
            }
            
            if(src.width() != dst->width() || src.height() != dst->height()) return HPC_INVALID_ARGS;

            if( src.ptr() == guided.ptr() && bt == BORDER_DEFAULT ) {
                func_fastGuidedFilterReflect101_SameMat<T, ncSrc, ncDst, nc>(src, dst, eps, scale);
            } else if(src.ptr() != guided.ptr() && bt == BORDER_DEFAULT) {
                func_fastGuidedFilterReflect101<T, ncSrc, ncDst, nc>(guided, src, dst, radius, eps, scale);
            } else {
                return HPC_NOT_SUPPORTED;
            }

            return HPC_SUCCESS;
        }

#if defined(FASTCV_USE_X86)
    template HPCStatus_t fastGuidedFilter<BORDER_DEFAULT, float, 1, 1, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& guided, const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *dst, int radius, float eps, int scale);
#endif

#if defined(FASTCV_USE_ARM)
    template HPCStatus_t fastGuidedFilter<BORDER_DEFAULT, float, 1, 1, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& guided, const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *dst, int radius, float eps, int scale);
#endif

#if defined(FASTCV_USE_CUDA)
    template HPCStatus_t fastGuidedFilter<BORDER_DEFAULT, float, 1, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& guided, const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *dst, int radius, float eps, int scale);
#endif

#if defined(FASTCV_USE_OCL)
    template HPCStatus_t fastGuidedFilter<BORDER_DEFAULT, float, 1, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& guided, const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *dst, int radius, float eps, int scale);
#endif
    //template HPCStatus_t fastGuidedFilter<float, 3, 3, 3, 3, BORDER_DEFAULT>(const Mat<float, 3>& guided, const Mat<float, 3>& src, Mat<float, 3> *dst, int radius, float eps, int scale);
    //template HPCStatus_t fastGuidedFilter<float, 4, 4, 4, 4, BORDER_DEFAULT>(const Mat<float, 4>& guided, const Mat<float, 4>& src, Mat<float, 4> *dst, int radius, float eps, int scale);


/*
    template HPCStatus_t boxFilter<uchar, 1, 1, 1, BORDER_DEFAULT>(const Mat<uchar, 1>& src, Size2i ks, Point2i anchor, Mat<uchar, 1> *dst, bool normalized);
    template HPCStatus_t boxFilter<uchar, 3, 3, 3, BORDER_DEFAULT>(const Mat<uchar, 3>& src, Size2i ks, Point2i anchor, Mat<uchar, 3> *dst, bool normalized);
    template HPCStatus_t boxFilter<uchar, 4, 4, 4, BORDER_DEFAULT>(const Mat<uchar, 4>& src, Size2i ks, Point2i anchor, Mat<uchar, 4> *dst, bool normalized);
*/

}}
