#include <status.h>
#include <vector.hpp>
#include <fastcv.hpp>

using namespace HPC::fmath;

namespace HPC{ namespace fastcv{
    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t add(const Mat<T, nc, type>& a, const Mat<T, nc, type>& b, Mat<T, nc, type> *c) {
            if(NULL == c->ptr()) return HPC_POINTER_NULL;
            if(a.ptr() == NULL) return HPC_POINTER_NULL;
            if(a.width() != b.width() && a.width() != c->width()) return HPC_INVALID_ARGS;
            if(a.height() != b.height() && a.height() != c->height()) return HPC_INVALID_ARGS;

            int height = a.height();
            int width = a.width();
            int astep = a.widthStep()/sizeof(T);
            int bstep = b.widthStep()/sizeof(T);
            int cstep = c->widthStep()/sizeof(T);

            size_t len = (size_t)width * nc;
            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                for(int i = 0; i < height; i++)
                    x86VectorAdd<T>(len, (T*)a.ptr() + i * astep, (T*)b.ptr() + i * bstep, (T*)c->ptr() + i * cstep);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                for(int i = 0; i < height; i++)
                    armVectorAdd<T>(len, (T*)a.ptr() + i * astep, (T*)b.ptr() + i * bstep, (T*)c->ptr() + i * cstep);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                for(int i = 0; i < height; i++)
                    cudaVectorAdd<T>(0, NULL, len, (T*)a.ptr() + i * astep, (T*)b.ptr() + i * bstep, (T*)c->ptr() + i * cstep);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                // oclVectorAdd<T>(queue, len, (cl_mem)a.ptr(), (cl_mem)b.ptr(), (cl_mem)c->ptr());
#endif
            }
            return HPC_SUCCESS;
        }
#if defined(FASTCV_USE_X86)
    template HPCStatus_t add<float, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& a, const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& b, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *c);
    template HPCStatus_t add<float, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& a, const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& b, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *c);
    template HPCStatus_t add<float, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& a, const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& b, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *c);
#endif

#if defined(FASTCV_USE_ARM)
    template HPCStatus_t add<float, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& a, const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& b, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *c);
    template HPCStatus_t add<float, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& a, const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& b, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *c);
    template HPCStatus_t add<float, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& a, const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& b, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *c);
#endif

#if defined(FASTCV_USE_CUDA)
    template HPCStatus_t add<float, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& a, const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& b, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *c);
    template HPCStatus_t add<float, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& a, const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& b, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *c);
    template HPCStatus_t add<float, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& a, const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& b, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *c);
#endif

#if defined(FASTCV_USE_OCL)
    template HPCStatus_t add<float, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& a, const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& b, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *c);
    template HPCStatus_t add<float, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& a, const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& b, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *c);
    template HPCStatus_t add<float, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& a, const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& b, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *c);
#endif

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t add(const Mat<T, nc, type>& a, T b, Mat<T, nc, type> *c) {
            if(NULL == c) return HPC_POINTER_NULL;
            if(a.width() != c->width()) return HPC_INVALID_ARGS;
            if(a.height() != c->height()) return HPC_INVALID_ARGS;

            int height = a.height();
            int width = a.width();
            int astep = a.widthStep()/sizeof(T);
            int cstep = c->widthStep()/sizeof(T);
            size_t len = (size_t)width * nc;
            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                for(int i = 0; i < height; i++)
                    x86VectorAddScalar<T>(len, (T*)a.ptr() + i * astep, b, (T*)c->ptr() + i * cstep);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                for(int i = 0; i < height; i++)
                    armVectorAddScalar<T>(len, (T*)a.ptr() + i * astep, b, (T*)c->ptr() + i * cstep);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                for(int i = 0; i < height; i++)
                    cudaVectorAddScalar<T>(0, NULL, len, (T*)a.ptr() + i * astep, b, (T*)c->ptr() + i * cstep);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                //   oclVectorAddScalar(queue, len, (cl_mem)a.ptr(), b, (cl_mem)c->ptr());
#endif
            }
            return HPC_SUCCESS;
        }
#if defined(FASTCV_USE_X86)
    template HPCStatus_t add<float, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& a, float b, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *c);
    template HPCStatus_t add<float, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& a, float b, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *c);
    template HPCStatus_t add<float, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& a, float b, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *c);
#endif

#if defined(FASTCV_USE_ARM)
    template HPCStatus_t add<float, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& a, float b, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *c);
    template HPCStatus_t add<float, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& a, float b, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *c);
    template HPCStatus_t add<float, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& a, float b, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *c);
#endif

#if defined(FASTCV_USE_CUDA)
    template HPCStatus_t add<float, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& a, float b, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *c);
    template HPCStatus_t add<float, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& a, float b, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *c);
    template HPCStatus_t add<float, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& a, float b, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *c);
#endif

#if defined(FASTCV_USE_OCL)
    template HPCStatus_t add<float, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& a, float b, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *c);
    template HPCStatus_t add<float, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& a, float b, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *c);
    template HPCStatus_t add<float, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& a, float b, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *c);
#endif

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t mul(const Mat<T, nc, type>& a, const Mat<T, nc, type>& b, Mat<T, nc, type> *c) {
            if(NULL == c) return HPC_POINTER_NULL;
            if(a.width() != b.width() && a.width() != c->width()) return HPC_INVALID_ARGS;
            if(a.height() != b.height() && a.height() != c->height()) return HPC_INVALID_ARGS;

            int height = a.height();
            int width = a.width();
            int astep = a.widthStep()/sizeof(T);
            int bstep = b.widthStep()/sizeof(T);
            int cstep = c->widthStep()/sizeof(T);

            size_t len = (size_t)width * nc;
            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                for(int i = 0; i < height; i++)
                    x86VectorMul<T>(len, (T*)a.ptr() + i * astep, (T*)b.ptr() + i * bstep, (T*)c->ptr() + i * cstep);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                for(int i = 0; i < height; i++)
                    armVectorMul<T>(len, (T*)a.ptr() + i * astep, (T*)b.ptr() + i * bstep, (T*)c->ptr() + i * cstep);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                for(int i = 0; i < height; i++)
                    cudaVectorMul<T>(0, NULL, len, (T*)a.ptr() + i * astep, (T*)b.ptr() + i * bstep, (T*)c->ptr() + i * cstep);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                //  oclVectorMul(queue, len, (cl_mem)a.ptr(), (cl_mem)b.ptr(), (cl_mem)c->ptr());
#endif
            }
            return HPC_SUCCESS;
        }
#if defined(FASTCV_USE_X86)
    template HPCStatus_t mul<float, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& a, const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& b, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *c);
    template HPCStatus_t mul<float, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& a, const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& b, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *c);
    template HPCStatus_t mul<float, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& a, const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& b, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *c);
#endif

#if defined(FASTCV_USE_ARM)
    template HPCStatus_t mul<float, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& a, const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& b, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *c);
    template HPCStatus_t mul<float, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& a, const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& b, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *c);
    template HPCStatus_t mul<float, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& a, const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& b, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *c);
#endif

#if defined(FASTCV_USE_CUDA)
    template HPCStatus_t mul<float, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& a, const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& b, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *c);
    template HPCStatus_t mul<float, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& a, const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& b, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *c);
    template HPCStatus_t mul<float, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& a, const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& b, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *c);
#endif

#if defined(FASTCV_USE_OCL)
    template HPCStatus_t mul<float, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& a, const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& b, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *c);
    template HPCStatus_t mul<float, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& a, const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& b, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *c);
    template HPCStatus_t mul<float, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& a, const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& b, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *c);
#endif

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t mul(const Mat<T, nc, type>& a, const T b, Mat<T, nc, type> *c) {
            if(NULL == c) return HPC_POINTER_NULL;
            if(a.width() != c->width()) return HPC_INVALID_ARGS;
            if(a.height() != c->height()) return HPC_INVALID_ARGS;

            int height = a.height();
            int width = a.width();
            int astep = a.widthStep()/sizeof(T);
            int cstep = c->widthStep()/sizeof(T);

            size_t len = (size_t)width * nc;
            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                for(int i = 0; i < height; i++)
                    x86VectorMulScalar<T>(len, (T*)a.ptr() + i * astep, b, (T*)c->ptr() + i * cstep);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                for(int i = 0; i < height; i++)
                    armVectorMulScalar<T>(len, (T*)a.ptr() + i * astep, b, (T*)c->ptr() + i * cstep);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                for(int i = 0; i < height; i++)
                    cudaVectorMulScalar<T>(0, NULL, len, (T*)a.ptr() + i * astep, b, (T*)c->ptr() + i * cstep);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                //   oclVectorMulScalar(queue, len, (cl_mem)a.ptr(), b, (cl_mem)c->ptr());
#endif
            }
            return HPC_SUCCESS;
        }
#if defined(FASTCV_USE_X86)
    template HPCStatus_t mul<float, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& a, const float b, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *c);
    template HPCStatus_t mul<float, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& a, const float b, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *c);
    template HPCStatus_t mul<float, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& a, const float b, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *c);
#endif

#if defined(FASTCV_USE_ARM)
    template HPCStatus_t mul<float, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& a, const float b, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *c);
    template HPCStatus_t mul<float, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& a, const float b, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *c);
    template HPCStatus_t mul<float, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& a, const float b, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *c);
#endif

#if defined(FASTCV_USE_CUDA)
    template HPCStatus_t mul<float, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& a, const float b, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *c);
    template HPCStatus_t mul<float, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& a, const float b, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *c);
    template HPCStatus_t mul<float, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& a, const float b, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *c);
#endif

#if defined(FASTCV_USE_OCL)
    template HPCStatus_t mul<float, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& a, const float b, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *c);
    template HPCStatus_t mul<float, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& a, const float b, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *c);
    template HPCStatus_t mul<float, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& a, const float b, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *c);
#endif

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t mls(const Mat<T, nc, type>& a, const Mat<T, nc, type>& b, Mat<T, nc, type> *c) {
            if(NULL == c) return HPC_POINTER_NULL;
            if(a.width() != b.width() && a.width() != c->width()) return HPC_INVALID_ARGS;
            if(a.height() != b.height() && a.height() != c->height()) return HPC_INVALID_ARGS;

            int height = a.height();
            int width = a.width();
            int astep = a.widthStep()/sizeof(T);
            int bstep = b.widthStep()/sizeof(T);
            int cstep = c->widthStep()/sizeof(T);

            size_t len = (size_t)width * nc;
            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                for(int i = 0; i < height; i++)
                    x86VectorMls<T>(len, (T*)a.ptr() + i * astep, (T*)b.ptr() + i * bstep, (T*)c->ptr() + i * cstep);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                for(int i = 0; i < height; i++)
                    armVectorMls<T>(len, (T*)a.ptr() + i * astep, (T*)b.ptr() + i * bstep, (T*)c->ptr() + i * cstep);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                for(int i = 0; i < height; i++)
                    cudaVectorMls<T>(0, NULL, len, (T*)a.ptr() + i * astep, (T*)b.ptr() + i * bstep, (T*)c->ptr() + i * cstep);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                //    oclVectorMls(queue, len, (cl_mem)a.ptr(), (cl_mem)b.ptr(), (cl_mem)c->ptr());
#endif
            }
            return HPC_SUCCESS;
        }
#if defined(FASTCV_USE_X86)
    template HPCStatus_t mls<float, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& a, const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& b, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *c);
    template HPCStatus_t mls<float, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& a, const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& b, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *c);
    template HPCStatus_t mls<float, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& a, const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& b, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *c);
#endif

#if defined(FASTCV_USE_ARM)
    template HPCStatus_t mls<float, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& a, const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& b, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *c);
    template HPCStatus_t mls<float, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& a, const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& b, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *c);
    template HPCStatus_t mls<float, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& a, const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& b, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *c);
#endif

#if defined(FASTCV_USE_CUDA)
    template HPCStatus_t mls<float, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& a, const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& b, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *c);
    template HPCStatus_t mls<float, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& a, const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& b, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *c);
    template HPCStatus_t mls<float, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& a, const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& b, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *c);
#endif

#if defined(FASTCV_USE_OCL)
    template HPCStatus_t mls<float, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& a, const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& b, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *c);
    template HPCStatus_t mls<float, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& a, const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& b, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *c);
    template HPCStatus_t mls<float, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& a, const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& b, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *c);
#endif

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t mla(const Mat<T, nc, type>& a, const Mat<T, nc, type>& b, Mat<T, nc, type> *c) {
            if(NULL == c) return HPC_POINTER_NULL;
            if(a.width() != b.width() && a.width() != c->width()) return HPC_INVALID_ARGS;
            if(a.height() != b.height() && a.height() != c->height()) return HPC_INVALID_ARGS;

            int height = a.height();
            int width = a.width();
            int astep = a.widthStep()/sizeof(T);
            int bstep = b.widthStep()/sizeof(T);
            int cstep = c->widthStep()/sizeof(T);

            size_t len = (size_t)width*nc;
            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                for(int i = 0; i < height; i++)
                    x86VectorMla<T>(len, (T*)a.ptr() + i * astep, (T*)b.ptr() + i * bstep, (T*)c->ptr() + i * cstep);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                for(int i = 0; i < height; i++)
                    armVectorMla<T>(len, (T*)a.ptr() + i * astep, (T*)b.ptr() + i * bstep, (T*)c->ptr() + i * cstep);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                for(int i = 0; i < height; i++)
                    cudaVectorMla<T>(0, NULL, len, (T*)a.ptr() + i * astep, (T*)b.ptr() + i * bstep, (T*)c->ptr() + i * cstep);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                //    oclVectorMla(queue, len, (cl_mem)a.ptr(), (cl_mem)b.ptr(), (cl_mem)c->ptr());
#endif
            }
            return HPC_SUCCESS;
        }
#if defined(FASTCV_USE_X86)
    template HPCStatus_t mla<float, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& a, const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& b, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *c);
    template HPCStatus_t mla<float, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& a, const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& b, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *c);
    template HPCStatus_t mla<float, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& a, const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& b, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *c);
#endif

#if defined(FASTCV_USE_ARM)
    template HPCStatus_t mla<float, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& a, const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& b, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *c);
    template HPCStatus_t mla<float, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& a, const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& b, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *c);
    template HPCStatus_t mla<float, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& a, const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& b, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *c);
#endif

#if defined(FASTCV_USE_CUDA)
    template HPCStatus_t mla<float, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& a, const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& b, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *c);
    template HPCStatus_t mla<float, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& a, const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& b, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *c);
    template HPCStatus_t mla<float, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& a, const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& b, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *c);
#endif

#if defined(FASTCV_USE_OCL)
    template HPCStatus_t mla<float, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& a, const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& b, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *c);
    template HPCStatus_t mla<float, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& a, const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& b, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *c);
    template HPCStatus_t mla<float, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& a, const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& b, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *c);
#endif

    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t div(const Mat<T, nc, type>& a, const Mat<T, nc, type>& b, Mat<T, nc, type> *c) {
            if(NULL == c) return HPC_POINTER_NULL;
            if(a.width() != b.width() && a.width() != c->width()) return HPC_INVALID_ARGS;
            if(a.height() != b.height() && a.height() != c->height()) return HPC_INVALID_ARGS;

            int height = a.height();
            int width = a.width();
            int astep = a.widthStep()/sizeof(T);
            int bstep = b.widthStep()/sizeof(T);
            int cstep = c->widthStep()/sizeof(T);

            size_t len = (size_t)width*nc;
            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                for(int i = 0; i < height; i++)
                    x86VectorDiv<T>(len, (T*)a.ptr() + i * astep, (T*)b.ptr() + i * bstep, (T*)c->ptr() + i * cstep);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                for(int i = 0; i < height; i++)
                    armVectorDiv<T>(len, (T*)a.ptr() + i * astep, (T*)b.ptr() + i * bstep, (T*)c->ptr() + i * cstep);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                for(int i = 0; i < height; i++)
                    cudaVectorDiv<T>(0, NULL, len, (T*)a.ptr() + i * astep, (T*)b.ptr() + i * bstep, (T*)c->ptr() + i * cstep);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                //     oclVectorDiv(queue, len, (cl_mem)a.ptr(), (cl_mem)b.ptr(), (cl_mem)c->ptr());
#endif
            }
            return HPC_SUCCESS;
        }

#if defined(FASTCV_USE_X86)
    template HPCStatus_t div<float, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& a, const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& b, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *c);
    template HPCStatus_t div<float, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& a, const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& b, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *c);
    template HPCStatus_t div<float, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& a, const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& b, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *c);
#endif

#if defined(FASTCV_USE_ARM)
    template HPCStatus_t div<float, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& a, const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& b, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *c);
    template HPCStatus_t div<float, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& a, const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& b, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *c);
    template HPCStatus_t div<float, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& a, const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& b, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *c);
#endif

#if defined(FASTCV_USE_CUDA)
    template HPCStatus_t div<float, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& a, const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& b, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *c);
    template HPCStatus_t div<float, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& a, const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& b, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *c);
    template HPCStatus_t div<float, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& a, const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& b, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *c);
#endif

#if defined(FASTCV_USE_OCL)
    template HPCStatus_t div<float, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& a, const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& b, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *c);
    template HPCStatus_t div<float, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& a, const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& b, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *c);
    template HPCStatus_t div<float, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& a, const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& b, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *c);
#endif
}};
