#if defined(FASTCV_USE_OCL)
#include"fastcv.hpp"
namespace HPC { namespace fastcv {
	Opencl opencl(PLATFORM_TYPE_NVIDIA,CL_DEVICE_TYPE_GPU,0);
}}
#endif
