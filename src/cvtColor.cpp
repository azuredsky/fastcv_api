#include <image.h>
#include <fastcv.hpp>
namespace HPC { namespace fastcv {
    template<ColorCvtType ct, typename Tsrc, int ncSrc, typename Tdst, int ncDst, EcoEnv_t type>
        HPCStatus_t cvtColor(const Mat<Tsrc, ncSrc, type>& src, Mat<Tdst, ncDst, type> *dst) {
            if(NULL == dst) return HPC_INVALID_ARGS;
            if(src.numDimensions() == 0 || dst->numDimensions() == 0) return HPC_NOT_INITIALIZED;
            int sw = src.width();
            int sh = src.height();
            int sstep = src.widthStep()/sizeof(Tsrc);

            int dw = dst->width();
            int dh = dst->height();
            int dstep = dst->widthStep()/sizeof(Tdst);

            void* sp = src.ptr();
            void* dp = dst->ptr();

            if(ct == BGR2NV21 || ct == BGR2NV12 || ct == RGB2NV21 || ct == BGR2I420  || sstep < sw*ncSrc || dstep < dw*ncDst|| ct == RGB2NV12) {
                if(sw % 2 != 0 || sh % 2 != 0 || sw != dw || sh != dh/3*2) return HPC_INVALID_ARGS;
            } else if(ct == NV212BGR || ct == NV122BGR || ct == NV212RGB || ct == I4202BGR || ct == YUV2GRAY_420|| ct == NV122RGB) {
                if(dw % 2 != 0 || dst->height() % 2 != 0 || sw != dw || sh/3*2 != dst->height() || sstep < sw*ncSrc || dstep < dw*ncDst) return HPC_INVALID_ARGS;

            }else{
                if(sw != dw || sh != dst->height() || sstep < sw*ncSrc || dstep < dw*ncDst) return HPC_INVALID_ARGS;
            }
            switch(ct) {
                case BGR2RGB:
                    {
                        if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                            x86BGR2RGBImage<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                            armBGR2RGBImage<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                            cudaBGR2RGBImage<Tsrc, ncSrc, Tdst, ncDst>(0, 0, sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                            oclBGR2RGBImage<Tsrc, ncSrc, Tdst, ncDst>(opencl.getCommandQueue(),
                                sh, sw,
                                sstep, (cl_mem)sp,
                                dstep, (cl_mem)dp);
#endif
                        }
                        break;
                    }
                case BGR2BGRA:
                    {
                        if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                            x86BGR2BGRAImage<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                            armBGR2BGRAImage<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                            cudaBGR2BGRAImage<Tsrc, ncSrc, Tdst, ncDst>(0, 0, sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                            oclBGR2BGRAImage<Tsrc, ncSrc, Tdst, ncDst>(
                                opencl.getCommandQueue(),
                                sh, sw,
                                sstep, (cl_mem)sp,
                                dstep, (cl_mem)dp);
#endif
                        }
                        break;
                    }
                case BGRA2BGR:
                    {
                        if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                            x86BGRA2BGRImage<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                            armBGRA2BGRImage<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                            cudaBGRA2BGRImage<Tsrc, ncSrc, Tdst, ncDst>(0, 0, sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                            oclBGRA2BGRImage<Tsrc, ncSrc, Tdst, ncDst>(
                                opencl.getCommandQueue(),
                                sh, sw,
                                sstep, (cl_mem)sp,
                                dstep, (cl_mem)dp);
#endif
                        }
                        break;
                    }
                case BGR2GRAY:
                    {
                        if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                            x86BGR2GRAYImage<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                            armBGR2GRAYImage<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                            cudaBGR2GRAYImage<Tsrc, ncSrc, Tdst, ncDst>(0, 0, sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                            oclBGR2GRAYImage<Tsrc, ncSrc, Tdst, ncDst>(
                                opencl.getCommandQueue(),
                                sh, sw,
                                sstep, (cl_mem)sp,
                                dstep, (cl_mem)dp);
#endif
                        }
                        break;
                    }
                case GRAY2BGR:
                    {
                        if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                            x86GRAY2BGRImage<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                            armGRAY2BGRImage<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                            cudaGRAY2BGRImage<Tsrc, ncSrc, Tdst, ncDst>(0, 0, sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                            oclGRAY2BGRImage<Tsrc, ncSrc, Tdst, ncDst>(
                                opencl.getCommandQueue(),
                                sh, sw,
                                sstep, (cl_mem)sp,
                                dstep, (cl_mem)dp);
#endif
                        }
                        break;
                    }
                case BGR2LAB:
                    {
                        if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                            x86BGR2LABImage<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                            armBGR2LABImage<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                            cudaBGR2LABImage<Tsrc, ncSrc, Tdst, ncDst>(0, 0, sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                            oclBGR2LABImage<Tsrc, ncSrc, Tdst, ncDst>(
                                opencl.getCommandQueue(),
                                sh, sw,
                                sstep, (cl_mem)sp,
                                dstep, (cl_mem)dp);
#endif
                        }
                        break;
                    }
                case LAB2BGR:
                    {
                        if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                            x86LAB2BGRImage<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                            armLAB2BGRImage<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                            cudaLAB2BGRImage<Tsrc, ncSrc, Tdst, ncDst>(0, 0, sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                            oclLAB2BGRImage<Tsrc, ncSrc, Tdst, ncDst>(
                                opencl.getCommandQueue(),
                                sh, sw,
                                sstep, (cl_mem)sp,
                                dstep, (cl_mem)dp);
#endif
                        }
                        break;
                    }
                case BGR2YCrCb:
                    {
                        if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                            x86BGR2YCrCbImage<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                            armBGR2YCrCbImage<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                            cudaBGR2YCrCbImage<Tsrc, ncSrc, Tdst, ncDst>(0, 0, sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                            oclBGR2YCrCbImage<Tsrc, ncSrc, Tdst, ncDst>(
                                opencl.getCommandQueue(),
                                sh, sw,
                                sstep, (cl_mem)sp,
                                dstep, (cl_mem)dp);
#endif
                        }
                        break;
                    }
                case YCrCb2BGR:
                    {
                        if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                            x86YCrCb2BGRImage<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                            armYCrCb2BGRImage<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                            cudaYCrCb2BGRImage<Tsrc, ncSrc, Tdst, ncDst>(0, 0, sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                            oclYCrCb2BGRImage<Tsrc, ncSrc, Tdst, ncDst>(
                                opencl.getCommandQueue(),
                                sh, sw,
                                sstep, (cl_mem)sp,
                                dstep, (cl_mem)dp);
#endif
                        }
                        break;
                    }
                case BGR2HSV:
                    {
                        if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                            x86BGR2HSVImage<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                            armBGR2HSVImage<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                            cudaBGR2HSVImage<Tsrc, ncSrc, Tdst, ncDst>(0, 0, sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                            oclBGR2HSVImage<Tsrc, ncSrc, Tdst, ncDst>(
                                opencl.getCommandQueue(),
                                sh, sw,
                                sstep, (cl_mem)sp,
                                dstep, (cl_mem)dp);
#endif
                        }
                        break;
                    }
                case HSV2BGR:
                    {
                        if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                            x86HSV2BGRImage<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                            armHSV2BGRImage<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                            cudaHSV2BGRImage<Tsrc, ncSrc, Tdst, ncDst>(0, 0, sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                            oclHSV2BGRImage<Tsrc, ncSrc, Tdst, ncDst>(
                                opencl.getCommandQueue(),
                                sh, sw,
                                sstep, (cl_mem)sp,
                                dstep, (cl_mem)dp);
#endif
                        }
                        break;
                    }
                case BGR2NV21:
                    {
                        if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                            x86BGR2NV21Image<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                            armBGR2NV21Image<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                            cudaBGR2NV21Image<Tsrc, ncSrc, Tdst, ncDst>(0, 0, sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                            oclBGR2NV21Image<Tsrc, ncSrc, Tdst, ncDst>(
                                opencl.getCommandQueue(),
                                sh, sw,
                                sstep, (cl_mem)sp,
                                dstep, (cl_mem)dp);
#endif
                        }
                        break;
                    }
                case NV212BGR:
                    {
                        if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                            x86NV212BGRImage<Tsrc, ncSrc, Tdst, ncDst>(dst->height(), sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                            armNV212BGRImage<Tsrc, ncSrc, Tdst, ncDst>(dst->height(), sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)

                            cudaNV212BGRImage<Tsrc, ncSrc, Tdst, ncDst>(0, 0, dst->height(), sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                            oclNV212BGRImage<Tsrc, ncSrc, Tdst, ncDst>(
                                opencl.getCommandQueue(),
                                dst->height(), sw,
                                sstep, (cl_mem)sp,
                                dstep, (cl_mem)dp);
#endif
                        }
                        break;
                    }
                case RGB2NV21:
                    {
                        if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                            x86RGB2NV21Image<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                            armRGB2NV21Image<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                            cudaRGB2NV21Image<Tsrc, ncSrc, Tdst, ncDst>(0, 0, sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                            oclRGB2NV21Image<Tsrc, ncSrc, Tdst, ncDst>(opencl.getCommandQueue(),
                                sh, sw,
                                sstep, (cl_mem)sp,
                                dstep, (cl_mem)dp);
#endif
                        }
                        break;
                    }
                case NV212RGB:
                    {
                        if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                            x86NV212RGBImage<Tsrc, ncSrc, Tdst, ncDst>(dst->height(), sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                            armNV212RGBImage<Tsrc, ncSrc, Tdst, ncDst>(dst->height(), sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                            cudaNV212RGBImage<Tsrc, ncSrc, Tdst, ncDst>(0, 0, dst->height(), sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                            oclNV212RGBImage<Tsrc, ncSrc, Tdst, ncDst>(opencl.getCommandQueue(),
                                dst->height(), sw,
                                sstep, (cl_mem)sp,
                                dstep, (cl_mem)dp);
#endif
                        }
                        break;
                    }

                case RGB2NV12:
                    {
                        if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                            x86RGB2NV12Image<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                            armRGB2NV12Image<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                            cudaRGB2NV12Image<Tsrc, ncSrc, Tdst, ncDst>(0, 0, sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                            /*      oclRGB2NV12Image<Tsrc, ncSrc, Tdst, ncDst>(opencl.getCommandQueue(),
                                    sh, sw,
                                    sstep, (cl_mem)sp,
                                    dstep, (cl_mem)dp);*/
#endif
                        }
                        break;
                    }
                case NV122RGB:
                    {
                        if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                            x86NV122RGBImage<Tsrc, ncSrc, Tdst, ncDst>(dst->height(), sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                            armNV122RGBImage<Tsrc, ncSrc, Tdst, ncDst>(dst->height(), sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                            cudaNV122RGBImage<Tsrc, ncSrc, Tdst, ncDst>(0, 0, dst->height(), sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                            /*      oclNV122RGBImage<Tsrc, ncSrc, Tdst, ncDst>(opencl.getCommandQueue(),
                                    dst->height(), sw,
                                    sstep, (cl_mem)sp,
                                    dstep, (cl_mem)dp);*/
#endif
                        }
                        break;
                    }


                case BGR2NV12:
                    {
                        if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                            x86BGR2NV12Image<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                            armBGR2NV12Image<Tsrc, ncSrc, Tdst, ncDst>(sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                            cudaBGR2NV12Image<Tsrc, ncSrc, Tdst, ncDst>(0, 0, sh, sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                            oclBGR2NV12Image<Tsrc, ncSrc, Tdst, ncDst>(opencl.getCommandQueue(),
                                sh, sw,
                                sstep, (cl_mem)sp,
                                dstep, (cl_mem)dp);
#endif
                        }
                        break;
                    }
                case NV122BGR:
                    {
                        if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                            x86NV122BGRImage<Tsrc, ncSrc, Tdst, ncDst>(dst->height(), sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                            armNV122BGRImage<Tsrc, ncSrc, Tdst, ncDst>(dst->height(), sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                            cudaNV122BGRImage<Tsrc, ncSrc, Tdst, ncDst>(0, 0, dst->height(), sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                            oclNV122BGRImage<Tsrc, ncSrc, Tdst, ncDst>(opencl.getCommandQueue(),
                                dst->height(), sw,
                                sstep, (cl_mem)sp,
                                dstep, (cl_mem)dp);
#endif
                        }
                        break;
                    }
                case I4202BGR:
                    {
                        if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                            x86I4202BGRImage<Tsrc, ncSrc, Tdst, ncDst>(dst->height(), sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                            armI4202BGRImage<Tsrc, ncSrc, Tdst, ncDst>(dst->height(), sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                            cudaI4202BGRImage<Tsrc, ncSrc, Tdst, ncDst>(0, 0, dst->height(), sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                            /*              oclI4202BGRImage<Tsrc, ncSrc, Tdst, ncDst>(opencl.getCommandQueue(),
                                            dst->height(), sw,
                                            sstep, (cl_mem)sp,
                                            dstep, (cl_mem)dp);*/
#endif
                        }
                        break;
                    }
                case BGR2I420:
                    {
                        if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                            x86BGR2I420Image<Tsrc, ncSrc, Tdst, ncDst>(src.height(), sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                            armBGR2I420Image<Tsrc, ncSrc, Tdst, ncDst>(src.height(), sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                            cudaBGR2I420Image<Tsrc, ncSrc, Tdst, ncDst>(0, 0, src.height(), sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                            /*          oclBGR2I420Image<Tsrc, ncSrc, Tdst, ncDst>(opencl.getCommandQueue(),
                                        src.height(), sw,
                                        sstep, (cl_mem)sp,
                                        dstep, (cl_mem)dp);*/
#endif
                        }
                        break;
                    }
                case YUV2GRAY_420:
                    {
                        if(type == EcoEnv_t::ECO_ENV_X86) {
#if defined(FASTCV_USE_X86)
                            x86YUV2GRAYImage<Tsrc, ncSrc, Tdst, ncDst>(dst->height(), sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_ARM) {
#if defined(FASTCV_USE_ARM)
                            armYUV2GRAYImage<Tsrc, ncSrc, Tdst, ncDst>(dst->height(), sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_CUDA) {
#if defined(FASTCV_USE_CUDA)
                            cudaYUV2GRAYImage<Tsrc, ncSrc, Tdst, ncDst>(0, 0, dst->height(), sw,
                                sstep, (const Tsrc*)sp,
                                dstep, (Tdst*)dp);
#endif
                        }
                        else if(type == EcoEnv_t::ECO_ENV_OCL) {
#if defined(FASTCV_USE_OCL)
                            /*             oclYUV2GRAYImage<Tsrc, ncSrc, Tdst, ncDst>(opencl.getCommandQueue(),
                                           dst->height(), sw,
                                           sstep, (cl_mem)sp,
                                           dstep, (cl_mem)dp);*/
#endif
                        }
                        break;
                    }

                default:
                    return HPC_NOT_SUPPORTED;
            }
            return HPC_SUCCESS;
        }

#if defined(FASTCV_USE_X86)
    //GRAY<->BGR(A)
    template HPCStatus_t cvtColor<GRAY2BGR, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<GRAY2BGR, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2GRAY, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2GRAY, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<GRAY2BGR, float, 1, float, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<GRAY2BGR, float, 1, float, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2GRAY, float, 3, float, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2GRAY, float, 4, float, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_X86> *dst);

    //BGR <--> BGRA
    template HPCStatus_t cvtColor<BGR2BGRA, uchar, 3, uchar, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGRA2BGR, uchar, 4, uchar, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2BGRA, float, 3, float, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGRA2BGR, float, 4, float, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);

    //BGR <--> RGB
    template HPCStatus_t cvtColor<BGR2RGB, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, uchar, 3, uchar, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, uchar, 4, uchar, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, uchar, 4, uchar, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, float, 3, float, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, float, 3, float, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, float, 4, float, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, float, 4, float, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *dst);

    // BGR <--> LAB
    template HPCStatus_t cvtColor<BGR2LAB, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2LAB, uchar, 4, uchar, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<LAB2BGR, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<LAB2BGR, uchar, 3, uchar, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2LAB, float, 3, float, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2LAB, float, 4, float, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<LAB2BGR, float, 3, float, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<LAB2BGR, float, 3, float, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *dst);

    // BGR <--> YCrCb
    template HPCStatus_t cvtColor<BGR2YCrCb, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2YCrCb, uchar, 4, uchar, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<YCrCb2BGR, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<YCrCb2BGR, uchar, 3, uchar, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2YCrCb, float, 3, float, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2YCrCb, float, 4, float, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<YCrCb2BGR, float, 3, float, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<YCrCb2BGR, float, 3, float, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *dst);

    // BGR <--> HSV
    template HPCStatus_t cvtColor<BGR2HSV, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2HSV, uchar, 4, uchar, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<HSV2BGR, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<HSV2BGR, uchar, 3, uchar, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2HSV, float, 3, float, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2HSV, float, 4, float, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<HSV2BGR, float, 3, float, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<HSV2BGR, float, 3, float, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<float, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_X86> *dst);

    // BGR <--> NV21
    template HPCStatus_t cvtColor<BGR2NV21, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2NV21, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<NV212BGR, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<NV212BGR, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst);

    // RGB <--> NV21
    template HPCStatus_t cvtColor<RGB2NV21, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<RGB2NV21, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<NV212RGB, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<NV212RGB, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst);

    // RGB <--> NV12
    template HPCStatus_t cvtColor<RGB2NV12, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<RGB2NV12, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<NV122RGB, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<NV122RGB, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst);


    // BGR <--> NV12
    template HPCStatus_t cvtColor<BGR2NV12, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2NV12, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<NV122BGR, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<NV122BGR, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst);


    // BGR <--> I420
    template HPCStatus_t cvtColor<BGR2I420, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<BGR2I420, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<I4202BGR, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_X86> *dst);
    template HPCStatus_t cvtColor<I4202BGR, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_X86> *dst);

    //YUV --> GRAY
    template HPCStatus_t cvtColor<YUV2GRAY_420, uchar, 1, uchar, 1, EcoEnv_t::ECO_ENV_X86>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_X86> *dst);
#endif

#if defined(FASTCV_USE_ARM)
    //GRAY<->BGR(A)
    template HPCStatus_t cvtColor<GRAY2BGR, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<GRAY2BGR, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2GRAY, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2GRAY, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<GRAY2BGR, float, 1, float, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<GRAY2BGR, float, 1, float, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2GRAY, float, 3, float, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2GRAY, float, 4, float, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_ARM> *dst);

    //BGR <--> BGRA
    template HPCStatus_t cvtColor<BGR2BGRA, uchar, 3, uchar, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGRA2BGR, uchar, 4, uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2BGRA, float, 3, float, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGRA2BGR, float, 4, float, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);

    //BGR <--> RGB
    template HPCStatus_t cvtColor<BGR2RGB, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, uchar, 3, uchar, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, uchar, 4, uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, uchar, 4, uchar, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, float, 3, float, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, float, 3, float, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, float, 4, float, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, float, 4, float, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *dst);

    // BGR <--> LAB
    template HPCStatus_t cvtColor<BGR2LAB, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2LAB, uchar, 4, uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<LAB2BGR, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<LAB2BGR, uchar, 3, uchar, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2LAB, float, 3, float, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2LAB, float, 4, float, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<LAB2BGR, float, 3, float, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<LAB2BGR, float, 3, float, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *dst);

    // BGR <--> YCrCb
    template HPCStatus_t cvtColor<BGR2YCrCb, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2YCrCb, uchar, 4, uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<YCrCb2BGR, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<YCrCb2BGR, uchar, 3, uchar, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2YCrCb, float, 3, float, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2YCrCb, float, 4, float, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<YCrCb2BGR, float, 3, float, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<YCrCb2BGR, float, 3, float, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *dst);

    // BGR <--> HSV
    template HPCStatus_t cvtColor<BGR2HSV, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2HSV, uchar, 4, uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<HSV2BGR, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<HSV2BGR, uchar, 3, uchar, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2HSV, float, 3, float, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2HSV, float, 4, float, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<HSV2BGR, float, 3, float, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<HSV2BGR, float, 3, float, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<float, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_ARM> *dst);

    // BGR <--> NV21
    template HPCStatus_t cvtColor<BGR2NV21, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2NV21, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<NV212BGR, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<NV212BGR, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst);

    // RGB <--> NV21
    template HPCStatus_t cvtColor<RGB2NV21, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<RGB2NV21, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<NV212RGB, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<NV212RGB, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst);

    // RGB <--> NV12
    template HPCStatus_t cvtColor<RGB2NV12, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<RGB2NV12, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<NV122RGB, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<NV122RGB, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst);


    // BGR <--> NV12
    template HPCStatus_t cvtColor<BGR2NV12, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2NV12, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<NV122BGR, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<NV122BGR, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst);


    // BGR <--> I420
    template HPCStatus_t cvtColor<BGR2I420, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<BGR2I420, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<I4202BGR, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_ARM> *dst);
    template HPCStatus_t cvtColor<I4202BGR, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_ARM> *dst);

    //YUV --> GRAY
    template HPCStatus_t cvtColor<YUV2GRAY_420, uchar, 1, uchar, 1, EcoEnv_t::ECO_ENV_ARM>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_ARM> *dst);
#endif

#if defined(FASTCV_USE_CUDA)
    //GRAY<->BGR(A)
    template HPCStatus_t cvtColor<GRAY2BGR, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<GRAY2BGR, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2GRAY, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2GRAY, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<GRAY2BGR, float, 1, float, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<GRAY2BGR, float, 1, float, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2GRAY, float, 3, float, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2GRAY, float, 4, float, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);

    template HPCStatus_t cvtColor<GRAY2BGR, half_t, 1, half_t, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<GRAY2BGR, half_t, 1, half_t, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2GRAY, half_t, 3, half_t, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2GRAY, half_t, 4, half_t, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);

    //BGR <--> BGRA
    template HPCStatus_t cvtColor<BGR2BGRA, uchar, 3, uchar, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGRA2BGR, uchar, 4, uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2BGRA, float, 3, float, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGRA2BGR, float, 4, float, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2BGRA, half_t, 3, half_t, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGRA2BGR, half_t, 4, half_t, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);

    //BGR <--> RGB
    template HPCStatus_t cvtColor<BGR2RGB, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, uchar, 3, uchar, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, uchar, 4, uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, uchar, 4, uchar, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, float, 3, float, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, float, 3, float, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, float, 4, float, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, float, 4, float, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, half_t, 3, half_t, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, half_t, 3, half_t, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, half_t, 4, half_t, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, half_t, 4, half_t, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    // BGR <--> LAB
    template HPCStatus_t cvtColor<BGR2LAB, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2LAB, uchar, 4, uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<LAB2BGR, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<LAB2BGR, uchar, 3, uchar, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2LAB, float, 3, float, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2LAB, float, 4, float, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<LAB2BGR, float, 3, float, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<LAB2BGR, float, 3, float, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2LAB, half_t, 3, half_t, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2LAB, half_t, 4, half_t, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<LAB2BGR, half_t, 3, half_t, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<LAB2BGR, half_t, 3, half_t, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    // BGR <--> YCrCb
    template HPCStatus_t cvtColor<BGR2YCrCb, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2YCrCb, uchar, 4, uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<YCrCb2BGR, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<YCrCb2BGR, uchar, 3, uchar, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2YCrCb, float, 3, float, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2YCrCb, float, 4, float, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<YCrCb2BGR, float, 3, float, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<YCrCb2BGR, float, 3, float, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2YCrCb, half_t, 3, half_t, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2YCrCb, half_t, 4, half_t, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<YCrCb2BGR, half_t, 3, half_t, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<YCrCb2BGR, half_t, 3, half_t, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    // BGR <--> HSV
    template HPCStatus_t cvtColor<BGR2HSV, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2HSV, uchar, 4, uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<HSV2BGR, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<HSV2BGR, uchar, 3, uchar, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2HSV, float, 3, float, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2HSV, float, 4, float, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<HSV2BGR, float, 3, float, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<HSV2BGR, float, 3, float, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<float, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2HSV, half_t, 3, half_t, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2HSV, half_t, 4, half_t, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<HSV2BGR, half_t, 3, half_t, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<HSV2BGR, half_t, 3, half_t, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<half_t, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<half_t, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    // BGR <--> NV21
    template HPCStatus_t cvtColor<BGR2NV21, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2NV21, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<NV212BGR, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<NV212BGR, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    // RGB <--> NV21
    template HPCStatus_t cvtColor<RGB2NV21, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<RGB2NV21, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<NV212RGB, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<NV212RGB, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    // RGB <--> NV12
    template HPCStatus_t cvtColor<RGB2NV12, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<RGB2NV12, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<NV122RGB, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<NV122RGB, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);


    // BGR <--> NV12
    template HPCStatus_t cvtColor<BGR2NV12, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2NV12, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<NV122BGR, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<NV122BGR, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);


    // BGR <--> I420
    template HPCStatus_t cvtColor<BGR2I420, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<BGR2I420, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<I4202BGR, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_CUDA> *dst);
    template HPCStatus_t cvtColor<I4202BGR, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_CUDA> *dst);

    //YUV --> GRAY
    template HPCStatus_t cvtColor<YUV2GRAY_420, uchar, 1, uchar, 1, EcoEnv_t::ECO_ENV_CUDA>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_CUDA> *dst);
#endif

#if defined(FASTCV_USE_OCL)
    //GRAY<->BGR(A)
    template HPCStatus_t cvtColor<GRAY2BGR, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<GRAY2BGR, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2GRAY, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2GRAY, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<GRAY2BGR, float, 1, float, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<GRAY2BGR, float, 1, float, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2GRAY, float, 3, float, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2GRAY, float, 4, float, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 1, EcoEnv_t::ECO_ENV_OCL> *dst);

    template HPCStatus_t cvtColor<GRAY2BGR, cl_half, 1, cl_half, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<GRAY2BGR, cl_half, 1, cl_half, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2GRAY, cl_half, 3, cl_half, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2GRAY, cl_half, 4, cl_half, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 1, EcoEnv_t::ECO_ENV_OCL> *dst);

    //BGR <--> BGRA
    template HPCStatus_t cvtColor<BGR2BGRA, uchar, 3, uchar, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGRA2BGR, uchar, 4, uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2BGRA, float, 3, float, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGRA2BGR, float, 4, float, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2BGRA, cl_half, 3, cl_half, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGRA2BGR, cl_half, 4, cl_half, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);

    //BGR <--> RGB
    template HPCStatus_t cvtColor<BGR2RGB, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, uchar, 3, uchar, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, uchar, 4, uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, uchar, 4, uchar, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, float, 3, float, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, float, 3, float, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, float, 4, float, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, float, 4, float, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, cl_half, 3, cl_half, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, cl_half, 3, cl_half, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, cl_half, 4, cl_half, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2RGB, cl_half, 4, cl_half, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    // BGR <--> LAB
    template HPCStatus_t cvtColor<BGR2LAB, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2LAB, uchar, 4, uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<LAB2BGR, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<LAB2BGR, uchar, 3, uchar, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2LAB, float, 3, float, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2LAB, float, 4, float, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<LAB2BGR, float, 3, float, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<LAB2BGR, float, 3, float, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2LAB, cl_half, 3, cl_half, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2LAB, cl_half, 4, cl_half, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<LAB2BGR, cl_half, 3, cl_half, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<LAB2BGR, cl_half, 3, cl_half, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    // BGR <--> YCrCb
    template HPCStatus_t cvtColor<BGR2YCrCb, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2YCrCb, uchar, 4, uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<YCrCb2BGR, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<YCrCb2BGR, uchar, 3, uchar, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2YCrCb, float, 3, float, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2YCrCb, float, 4, float, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<YCrCb2BGR, float, 3, float, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<YCrCb2BGR, float, 3, float, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2YCrCb, cl_half, 3, cl_half, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2YCrCb, cl_half, 4, cl_half, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<YCrCb2BGR, cl_half, 3, cl_half, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<YCrCb2BGR, cl_half, 3, cl_half, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    // BGR <--> HSV
    template HPCStatus_t cvtColor<BGR2HSV, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2HSV, uchar, 4, uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<HSV2BGR, uchar, 3, uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<HSV2BGR, uchar, 3, uchar, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2HSV, float, 3, float, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2HSV, float, 4, float, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<HSV2BGR, float, 3, float, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<HSV2BGR, float, 3, float, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<float, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<float, 4, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2HSV, cl_half, 3, cl_half, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2HSV, cl_half, 4, cl_half, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<HSV2BGR, cl_half, 3, cl_half, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<HSV2BGR, cl_half, 3, cl_half, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<cl_half, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<cl_half, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    // BGR <--> NV21
    template HPCStatus_t cvtColor<BGR2NV21, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2NV21, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<NV212BGR, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<NV212BGR, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    // RGB <--> NV21
    template HPCStatus_t cvtColor<RGB2NV21, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<RGB2NV21, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<NV212RGB, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<NV212RGB, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    // RGB <--> NV12
    template HPCStatus_t cvtColor<RGB2NV12, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<RGB2NV12, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<NV122RGB, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<NV122RGB, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst);


    // BGR <--> NV12
    template HPCStatus_t cvtColor<BGR2NV12, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2NV12, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<NV122BGR, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<NV122BGR, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst);


    // BGR <--> I420
    template HPCStatus_t cvtColor<BGR2I420, uchar, 3, uchar, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<BGR2I420, uchar, 4, uchar, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<I4202BGR, uchar, 1, uchar, 3, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 3, EcoEnv_t::ECO_ENV_OCL> *dst);
    template HPCStatus_t cvtColor<I4202BGR, uchar, 1, uchar, 4, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 4, EcoEnv_t::ECO_ENV_OCL> *dst);

    //YUV --> GRAY
    template HPCStatus_t cvtColor<YUV2GRAY_420, uchar, 1, uchar, 1, EcoEnv_t::ECO_ENV_OCL>(const Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL>& src, Mat<uchar, 1, EcoEnv_t::ECO_ENV_OCL> *dst);
#endif
}}
