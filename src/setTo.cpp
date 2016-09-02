#include <status.h>
#include <fastcv.hpp>
#include <vector.hpp>
using namespace HPC::fmath;

namespace HPC { namespace fastcv {
    template<typename T, int nc, EcoEnv_t type> HPCStatus_t setTo(T value, Mat<T, nc, type> *mat) {
            if(NULL == mat->ptr()) return HPC_POINTER_NULL;

            int height = mat->height();
            int width = mat->width();
            int step = mat->widthStep()/sizeof(T);

            size_t len = (size_t)width * nc;
            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                for(int i = 0; i < height; i++)
                    x86VectorFillScalar<T>(len, value, (T*)mat->ptr() + i * step);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                for(int i = 0; i < height; i++)
                    armVectorFillScalar<T>(len, value, (T*)mat->ptr() + i * step);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                // cudaVectorFillScalar<T>(0, 0, nc*mat->width()*mat->height(), value, (T*)mat->ptr());
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                //  oclVectorFillScalar<T>(queue, nc*mat->width()*mat->height(), value, (cl_mem)mat->ptr());
#endif
            }
            return HPC_SUCCESS;
        }
#if defined(FASTCV_USE_X86)
    template HPCStatus_t setTo<uchar, 1, EcoEnv_t::ECO_ENV_X86>(const uchar value, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *mat);
    template HPCStatus_t setTo<uchar, 3, EcoEnv_t::ECO_ENV_X86>(const uchar value, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *mat);
    template HPCStatus_t setTo<uchar, 4, EcoEnv_t::ECO_ENV_X86>(const uchar value, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *mat);

    template HPCStatus_t setTo<float, 1, EcoEnv_t::ECO_ENV_X86>(const float value, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *mat);
    template HPCStatus_t setTo<float, 3, EcoEnv_t::ECO_ENV_X86>(const float value, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *mat);
    template HPCStatus_t setTo<float, 4, EcoEnv_t::ECO_ENV_X86>(const float value, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *mat);
#endif

#if defined(FASTCV_USE_ARM)
    template HPCStatus_t setTo<uchar, 1, EcoEnv_t::ECO_ENV_ARM>(const uchar value, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *mat);
    template HPCStatus_t setTo<uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const uchar value, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *mat);
    template HPCStatus_t setTo<uchar, 4, EcoEnv_t::ECO_ENV_ARM>(const uchar value, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *mat);

    template HPCStatus_t setTo<float, 1, EcoEnv_t::ECO_ENV_ARM>(const float value, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *mat);
    template HPCStatus_t setTo<float, 3, EcoEnv_t::ECO_ENV_ARM>(const float value, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *mat);
    template HPCStatus_t setTo<float, 4, EcoEnv_t::ECO_ENV_ARM>(const float value, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *mat);
#endif

#if defined(FASTCV_USE_CUDA)
    template HPCStatus_t setTo<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>(const uchar value, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *mat);
    template HPCStatus_t setTo<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const uchar value, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *mat);
    template HPCStatus_t setTo<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>(const uchar value, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *mat);

    template HPCStatus_t setTo<float, 1, EcoEnv_t::ECO_ENV_CUDA>(const float value, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *mat);
    template HPCStatus_t setTo<float, 3, EcoEnv_t::ECO_ENV_CUDA>(const float value, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *mat);
    template HPCStatus_t setTo<float, 4, EcoEnv_t::ECO_ENV_CUDA>(const float value, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *mat);
#endif

#if defined(FASTCV_USE_OCL)
    template HPCStatus_t setTo<uchar, 1, EcoEnv_t::ECO_ENV_OCL>(const uchar value, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *mat);
    template HPCStatus_t setTo<uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const uchar value, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *mat);
    template HPCStatus_t setTo<uchar, 4, EcoEnv_t::ECO_ENV_OCL>(const uchar value, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *mat);

    template HPCStatus_t setTo<float, 1, EcoEnv_t::ECO_ENV_OCL>(const float value, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *mat);
    template HPCStatus_t setTo<float, 3, EcoEnv_t::ECO_ENV_OCL>(const float value, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *mat);
    template HPCStatus_t setTo<float, 4, EcoEnv_t::ECO_ENV_OCL>(const float value, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *mat);
#endif

    template<typename T, int nc, EcoEnv_t type> HPCStatus_t setTo(const T value[nc], Mat<T, nc, type> *mat) {
            if(NULL == mat) return HPC_POINTER_NULL;
            int height = mat->height();
            int width = mat->width();
            int step = mat->widthStep()/sizeof(T);

            size_t len = (size_t)width * nc;
            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                for(int i = 0; i < height; i++)
                    x86VectorFillVector<T, nc>(len, value, (T*)mat->ptr() + i * step);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                for(int i = 0; i < height; i++)
                    armVectorFillVector<T, nc>(len, value, (T*)mat->ptr() + i * step);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                //    cudaVectorFillVector<T, nc>(0, 0, mat->height()*mat->width(), value, (T*)mat->ptr());
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                //  oclVectorFillVector<T, nc>(queue, mat->height()*mat->width(), value, (cl_mem)mat->ptr());
#endif
            }
            return HPC_SUCCESS;
        }

#if defined(FASTCV_USE_X86)
    template HPCStatus_t setTo<float, 3, EcoEnv_t::ECO_ENV_X86>(const float value[3], Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *mat);
#endif

#if defined(FASTCV_USE_ARM)
    template HPCStatus_t setTo<float, 3, EcoEnv_t::ECO_ENV_ARM>(const float value[3], Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *mat);
#endif

#if defined(FASTCV_USE_CUDA)
    template HPCStatus_t setTo<float, 3, EcoEnv_t::ECO_ENV_CUDA>(const float value[3], Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *mat);
#endif

#if defined(FASTCV_USE_OCL)
    template HPCStatus_t setTo<float, 3, EcoEnv_t::ECO_ENV_OCL>(const float value[3], Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *mat);
#endif
}};
