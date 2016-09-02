#include <fastcv.hpp>
#include <image.h>
//#include "func_inline.h"
#include<stdio.h>

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

    template<typename T, int ncSrc, int ncDst, int nc, EcoEnv_t type>
    inline HPCStatus_t func_fastGuidedFilterReflect101_SameMat(const Mat<T, ncSrc, type>& src, Mat<T, ncDst, type> *dst, float eps, int scale);

    template<typename T, int ncSrc, int ncDst, int nc, EcoEnv_t type>
    inline HPCStatus_t func_fastGuidedFilterReflect101_SameMat(const Mat<T, ncSrc, type>& src, Mat<T, ncDst, type> *dst, float eps, int scale) {

        HPCStatus_t hst;
        int H = src.height(), W = src.width();
        int H_sub = H/scale;
        int W_sub = W/scale;
        Size size_sub(W_sub, H_sub);
        Size size(W, H);

        Mat<T, nc, type> src_sub(size_sub);
        hst = resize<INTERPOLATION_TYPE_LINEAR, T, nc, T, nc, nc>(src, &src_sub);
        hst = resize<INTERPOLATION_TYPE_LINEAR, T, nc, T, nc, nc>(src, &src_sub);

        Size ksize(3, 3);
        Mat<T, nc, type> Ip(src_sub.height(), src_sub.width());
        hst = mul<T, nc>(src_sub, src_sub, &Ip);

        Mat<T, nc, type> mean_I(H_sub, W_sub);
        Mat<T, nc, type> cov_Ip(H_sub, W_sub);
        hst = boxFilter<BORDER_DEFAULT, T, nc>(src_sub, ksize, &mean_I, true);

 
        hst = boxFilter<BORDER_DEFAULT, T, nc>(Ip, ksize, &cov_Ip, true);

        hst = mls<T, nc>(mean_I, mean_I, &cov_Ip);
        Mat<T, nc, type> a_sub(H_sub, W_sub);
        Mat<T, nc, type> a_sub_tmp(H_sub, W_sub);
        Mat<T, nc, type> b_sub(H_sub, W_sub);

        add<T, nc>(cov_Ip, eps, &a_sub);
        div<T, nc>(cov_Ip, a_sub, &a_sub_tmp);
        mls<T, nc>(mean_I, a_sub_tmp, &mean_I);

        boxFilter<BORDER_DEFAULT, T, nc>(a_sub_tmp, ksize, &a_sub, true);
        boxFilter<BORDER_DEFAULT, T, nc>(mean_I, ksize, &b_sub, true);

        Mat<T, nc, type> a(H, W);
        //Mat<T, nc> b(H, W);
        resize<INTERPOLATION_TYPE_LINEAR, T, nc, T, nc, nc>(a_sub, &a);
        resize<INTERPOLATION_TYPE_LINEAR, T, nc, T, nc, nc>(b_sub, dst);
        mla<float, 1>(src, a, dst);

        return HPC_SUCCESS;
    }

    template<typename T, int ncSrc, int ncDst, int nc, EcoEnv_t type>
    inline HPCStatus_t func_fastGuidedFilterReflect101 (const Mat<T, ncSrc, type>& guided, const Mat<T, ncSrc, type>& src, Mat<T, ncDst, type> *dst, int radius, float eps, int scale);

    template<typename T, int ncSrc, int ncDst, int nc, EcoEnv_t type>
    inline HPCStatus_t func_fastGuidedFilterReflect101 (const Mat<T, ncSrc, type>& guided, const Mat<T, ncSrc, type>& src, Mat<T, ncDst, type> *dst, int radius, float eps, int scale) {
        if( radius < scale) radius = scale;

        HPCStatus_t hst;
        int H = guided.height(), W = guided.width();
        int H_sub = H/scale;
        int W_sub = W/scale;
        int r_sub = radius / scale;
        Size size_sub(W_sub, H_sub);
        Size size(W, H);

        Mat<T, nc, type> guided_sub(size_sub);
        Mat<T, nc, type> src_sub(size_sub);

        hst = resize<INTERPOLATION_TYPE_LINEAR, T, nc, T, nc, nc>(guided, &guided_sub);
        hst = resize<INTERPOLATION_TYPE_LINEAR, T, nc, T, nc, nc>(src, &src_sub);
        Size ksize(r_sub*2+1, r_sub*2+1);
        Mat<T, nc, type> II(src_sub.height(), src_sub.width());
        Mat<T, nc, type> Ip(src_sub.height(), src_sub.width());

        hst = mul<T, nc>(guided_sub, guided_sub, &II);
        hst = mul<T, nc>(guided_sub, src_sub, &Ip);

        Mat<T, nc, type> mean_I(H_sub, W_sub);
        hst = boxFilter<BORDER_DEFAULT, T, nc>(guided_sub, ksize, &mean_I, true);

        Mat<T, nc, type> mean_p(H_sub, W_sub);
        hst = boxFilter<BORDER_DEFAULT, T, nc>(src_sub, ksize, &mean_p, true);

        Mat<T, nc,type> cov_Ip(H_sub, W_sub);
        hst = boxFilter<BORDER_DEFAULT, T, nc>(Ip, ksize, &cov_Ip, true);
        hst = mls<T, nc>(mean_I, mean_p, &cov_Ip);

        Mat<T, nc,type> var_I(H_sub, W_sub);
        hst = boxFilter<BORDER_DEFAULT, T, nc>(II, ksize, &var_I, true);
        hst = mls<T, nc>(mean_I, mean_I, &var_I);

        Mat<T, nc,type> a_sub(H_sub, W_sub);
        Mat<T, nc,type> a_sub_tmp(H_sub, W_sub);
        Mat<T, nc,type> b_sub(H_sub, W_sub);

        add<T, nc>(var_I, eps, &a_sub);
        div<T, nc>(cov_Ip, a_sub, &a_sub_tmp);
        mls<T, nc>(mean_I, a_sub_tmp, &mean_p);

        boxFilter<BORDER_DEFAULT, T, nc>(a_sub_tmp, ksize, &a_sub, true);
        boxFilter<BORDER_DEFAULT, T, nc>(mean_p, ksize, &b_sub, true);

        Mat<T, nc, type> a(H, W);
        resize<INTERPOLATION_TYPE_LINEAR, T, nc, T, nc, nc>(a_sub, &a);
        resize<INTERPOLATION_TYPE_LINEAR, T, nc, T, nc, nc>(b_sub, dst);
        mla<float, 1>(guided, a, dst);
        return HPC_SUCCESS;
    }




#if defined(FASTCV_USE_X86)
    template HPCStatus_t func_fastGuidedFilterReflect101_SameMat<float, 1, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *dst, float eps, int scale);
#endif

#if defined(FASTCV_USE_ARM)
    template HPCStatus_t func_fastGuidedFilterReflect101_SameMat<float, 1, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *dst, float eps, int scale);
#endif

#if defined(FASTCV_USE_CUDA)
    template HPCStatus_t func_fastGuidedFilterReflect101_SameMat<float, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *dst, float eps, int scale);
#endif

#if defined(FASTCV_USE_OCL)
    template HPCStatus_t func_fastGuidedFilterReflect101_SameMat<float, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *dst, float eps, int scale);
#endif

#if defined(FASTCV_USE_X86)
    template HPCStatus_t func_fastGuidedFilterReflect101<float, 1, 1, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& gided, const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *dst, int radius, float eps, int scale);
#endif

#if defined(FASTCV_USE_ARM)
    template HPCStatus_t func_fastGuidedFilterReflect101<float, 1, 1, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& gided, const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *dst, int radius, float eps, int scale);
#endif

#if defined(FASTCV_USE_CUDA)
    template HPCStatus_t func_fastGuidedFilterReflect101<float, 1, 1, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& gided, const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *dst, int radius, float eps, int scale);
#endif

#if defined(FASTCV_USE_OCL)
    template HPCStatus_t func_fastGuidedFilterReflect101<float, 1, 1, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& gided, const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *dst, int radius, float eps, int scale);
#endif


}}
