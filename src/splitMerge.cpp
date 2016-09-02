#include <status.h>
#include <image.h>
#include <fastcv.hpp>

namespace HPC { namespace fastcv {
    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t merge(const Mat<T, 1, type> *src, Mat<T, nc, type> *dst) {
            if(src->ptr() == NULL) return HPC_POINTER_NULL;
            if(dst->ptr() == NULL) return HPC_POINTER_NULL;
            
            if(src->numDimensions() == 0 || dst->numDimensions() == 0) return HPC_NOT_INITIALIZED;
            int sw = src->width();
            int sh = src->height();
            int sstep = src->widthStep()/sizeof(T);

            int dw = dst->width();
            int dh = dst->height();
            int dstep = dst->widthStep()/sizeof(T);
            //T* dp = dst->ptr();
            void* dp = dst->ptr();
            for(int i = 0; i < nc; i++) {
                if((src+i)->width() != dw) return HPC_INVALID_ARGS;
                if((src+i)->height() != dh) return HPC_INVALID_ARGS;
            }

            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                const T* inData[nc];
                for(int i = 0; i < nc; i++) {
                    inData[i] = (T*)src[i].ptr();
                }
                x86MergeNChannelsImage<T, nc>(sh, sw,
                    sstep, inData,
                    dstep, (T*)dp);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                const T* inData[nc];
                for(int i = 0; i < nc; i++) {
                    inData[i] = (T*)src[i].ptr();
                }
                armMergeNChannelsImage<T, nc>(src->height(), sw,
                    sstep, inData,
                    dstep, (T*)dp);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                const T* inData[nc];
                for(int i = 0; i < nc; i++) {
                    inData[i] = (T*)src[i].ptr();
                }
                cudaMergeNChannelsImage<T, nc>(0, NULL,
                    src->height(), sw,
                    sstep, inData,
                    dstep, (T*)dp);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                //    cl_mem inData[nc];
                //for(int i = 0; i < nc; i++) {
                //    inData[i] = (cl_mem*)src[i].ptr();
                //}
                //    oclMergeNChannelsImage<T, nc>(queue,
                //             src->height(), sw,
                //             sw, inData,
                //             dw, (cl_mem)dp);
#endif
            }
            return HPC_SUCCESS;
        }
#if defined(FASTCV_USE_X86)
    template HPCStatus_t merge<uchar, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t merge<float, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *src, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t merge<float, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *src, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *dst);
#endif

#if defined(FASTCV_USE_ARM)
    template HPCStatus_t merge<uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t merge<float, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *src, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t merge<float, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *src, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *dst);
#endif

#if defined(FASTCV_USE_CUDA)
    template HPCStatus_t merge<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t merge<float, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *src, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t merge<float, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *src, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
#endif

#if defined(FASTCV_USE_OCL)
    template HPCStatus_t merge<uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t merge<float, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *src, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t merge<float, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *src, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
#endif

    /*split support contiguous memory and uncontinous memory storage
     when dst->height() == src.height(), it is the uncontinous memory storage,
     when dst->height() == src.height() * nc, it is the continous memory storage. */
    template<typename T, int nc, EcoEnv_t type>
        HPCStatus_t split(const Mat<T, nc, type>& src, Mat<T, 1, type> *dst) {

            if(NULL == dst) return HPC_POINTER_NULL;
            int sw = src.width();
            int sh = src.height();
            if(dst[0].height() != sh|| dst[0].height() != sh * 3)
                return HPC_INVALID_ARGS;
            int sstep = src.widthStep()/sizeof(T);

            int dstep = dst->widthStep()/sizeof(T);

            //const T* sp = src.ptr();
			void* sp = src.ptr();
            int dw = dst->width();
            if(sh == dst[0].height()){
                for(int i = 0; i < nc; i++) {
                    if(sw != (dst+i)->width()) return HPC_INVALID_ARGS;
                    if(sh != (dst+i)->height()) return HPC_INVALID_ARGS;
                }
            }

            if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                T* outData[nc];
                if(dst[0].height() == sh){
                    for(int i = 0; i < nc; i++) {
                        outData[i] = (T*)(dst+i)->ptr();
                    }
                } else if(dst[0].height() == sh * 3){
                    for(int i = 0; i < nc; i++) {
                        outData[i] = (T*)dst->ptr() + sh * dstep * i;
                    }
                }
                x86SplitNChannelsImage<T, nc>(sh, sw,
                    sstep, (T*)sp,
                    dstep, outData);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                T* outData[nc];
                if(dst[0].height() == sh){
                    for(int i = 0; i < nc; i++) {
                        outData[i] = (T*)(dst+i)->ptr();
                    }
                } else if(dst[0].height() == sh * 3){
                    for(int i = 0; i < nc; i++) {
                        outData[i] = (T*)dst->ptr() + sh * dstep * i;
                    }
                }
                armSplitNChannelsImage<T, nc>(sh, sw,
                    sstep, (T*)sp,
                    dstep, outData);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                T* outData[nc];
                if(dst[0].height() == sh){
                    for(int i = 0; i < nc; i++) {
                        outData[i] = (T*)(dst+i)->ptr();
                    }
                } else if(dst[0].height() == sh * 3){
                    for(int i = 0; i < nc; i++) {
                        outData[i] = (T*)dst->ptr() + sh * dstep * i;
                    }
                }
                cudaSplitNChannelsImage<T, nc>(0, NULL,
                    sh, sw,
                    sstep, (T*)sp,
                    dstep, outData);
#endif
            }
            else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                //   cl_mem outData[nc];
                //if(dst[0].height() == sh){
                //   for(int i = 0; i < nc; i++) {
                //       outData[i] = (cl_mem)(dst+i)->ptr();
                //   }
                //} else if(dst[0].height() == sh * 3){
                //   for(int i = 0; i < nc; i++) {
                //        outData[i] = (cl_mem)dst->ptr() + sh * dstep * i;
                //   }
                //}
                //   oclSplitNChannelsImage<T, nc>(queue,
                //            sh, sw,
                //            sw, (cl_mem)sp,
                //            dw, outData);
#endif
            }
            return HPC_SUCCESS;
        }
#if defined(FASTCV_USE_X86)
    template HPCStatus_t split<uchar, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t split<float, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t split<float, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *dst);
#endif

#if defined(FASTCV_USE_ARM)
    template HPCStatus_t split<uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t split<float, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t split<float, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
#endif

#if defined(FASTCV_USE_CUDA)
    template HPCStatus_t split<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t split<float, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t split<float, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
#endif

#if defined(FASTCV_USE_OCL)
    template HPCStatus_t split<uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t split<float, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t split<float, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
#endif
}};
