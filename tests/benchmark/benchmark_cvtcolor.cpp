#include <fastcv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#if defined(USE_OCL)
#include "opencv2/ocl/ocl.hpp"
#endif
#if defined(USE_CUDA)
#include "opencv2/gpu/gpu.hpp"
#endif
#include <sys/time.h>
#include "mytime.h"
using namespace HPC::fastcv;

template<typename T>
void randomRangeData(T *data, const int max, const size_t num){
    size_t tmp;
    clock_t ct = clock();
    srand((unsigned int)ct);

    for(size_t i = 0; i < num; i++){
        tmp = rand()% max;
        data[i] = (T)tmp;
    }
}

inline void refrenshData(int inputWidth, int inputHeight,
    cv::Mat &o_u1_1,
    cv::Mat &o_u1_2,
    cv::Mat &o_f1_1,
    cv::Mat &o_f1_2,
    cv::Mat &o_u3_1,
    cv::Mat &o_u3_2,
    cv::Mat &o_f3_1,
    cv::Mat &o_f3_2,
    cv::Mat &o_u4_1,
    cv::Mat &o_u4_2,
    cv::Mat &o_f4_1,
    cv::Mat &o_f4_2,
    Mat<uchar, 1> &u1_1,
    Mat<uchar, 1> &u1_2,
    Mat<float, 1> &f1_1,
    Mat<float, 1> &f1_2,
    Mat<uchar, 3> &u3_1,
    Mat<uchar, 3> &u3_2,
    Mat<float, 3> &f3_1,
    Mat<float, 3> &f3_2,
    Mat<uchar, 4> &u4_1,
    Mat<uchar, 4> &u4_2,
    Mat<float, 4> &f4_1,
    Mat<float, 4> &f4_2){

    int inputNum = inputWidth * inputHeight;
    uchar* input_data_u1 = (uchar*)malloc(inputNum*sizeof(uchar));
    randomRangeData(input_data_u1, 255, inputNum);
    float* input_data_f1 = (float*)malloc(inputNum*sizeof(float));
    for(int i=0; i<inputNum; ++i)
        input_data_f1[i] = (float)input_data_u1[i]/255.f;

    uchar* input_data_u3 = (uchar*)malloc(inputNum*3*sizeof(uchar));
    randomRangeData(input_data_u3, 255, inputNum*3);
    float* input_data_f3 = (float*)malloc(inputNum*3*sizeof(float));
    for(int i=0; i<inputNum*3; ++i)
        input_data_f3[i] = (float)input_data_u3[i]/255.f;

    uchar* input_data_u4 = (uchar*)malloc(inputNum*4*sizeof(uchar));
    randomRangeData(input_data_u4, 255, inputNum*4);
    float* input_data_f4 = (float*)malloc(inputNum*4*sizeof(float));
    for(int i=0; i<inputNum*4; ++i)
        input_data_f4[i] = (float)input_data_u4[i]/255.f;

    u1_1.fromHost(input_data_u1);
    u1_2.fromHost(input_data_u1);
    f1_1.fromHost(input_data_f1);
    f1_2.fromHost(input_data_f1);
    u3_1.fromHost(input_data_u3);
    u3_2.fromHost(input_data_u3);
    f3_1.fromHost(input_data_f3);
    f3_2.fromHost(input_data_f3);
    u3_1.fromHost(input_data_u4);
    u3_2.fromHost(input_data_u4);
    f3_1.fromHost(input_data_f4);
    f3_2.fromHost(input_data_f4);

    memcpy(o_u1_1.ptr<uchar>(), input_data_u1, sizeof(uchar)*inputNum);
    memcpy(o_u1_2.ptr<uchar>(), input_data_u1, sizeof(uchar)*inputNum);
    memcpy(o_f1_1.ptr<float>(), input_data_f1, sizeof(float)*inputNum);
    memcpy(o_f1_2.ptr<float>(), input_data_f1, sizeof(float)*inputNum);
    memcpy(o_u3_1.ptr<uchar>(), input_data_u3, sizeof(uchar)*inputNum*3);
    memcpy(o_u3_2.ptr<uchar>(), input_data_u3, sizeof(uchar)*inputNum*3);
    memcpy(o_f3_1.ptr<float>(), input_data_f3, sizeof(float)*inputNum*3);
    memcpy(o_f3_2.ptr<float>(), input_data_f3, sizeof(float)*inputNum*3);
    memcpy(o_u4_1.ptr<uchar>(), input_data_u4, sizeof(uchar)*inputNum*4);
    memcpy(o_u4_2.ptr<uchar>(), input_data_u4, sizeof(uchar)*inputNum*4);
    memcpy(o_f4_1.ptr<float>(), input_data_f4, sizeof(float)*inputNum*4);
    memcpy(o_f4_2.ptr<float>(), input_data_f4, sizeof(float)*inputNum*4);

    free(input_data_u1);
    free(input_data_f1);
    free(input_data_u3);
    free(input_data_f3);
    free(input_data_u4);
    free(input_data_f4);
}

#define refrensh(){\
    refrenshData(inputWidth, inputHeight, o_u1_1, o_u1_2, o_f1_1, o_f1_2, o_u3_1, o_u3_2, o_f3_1, o_f3_2, o_u4_1, o_u4_2, o_f4_1, o_f4_2, u1_1, u1_2, f1_1, f1_2, u3_1, u3_2, f3_1, f3_2, u4_1, u4_2, f4_1, f4_2);\
}


void benchmark(int inputWidth, int inputHeight){
    //data prepare
#if defined(USE_ARM) || defined(USE_X86) || defined(USE_CUDA)
    //opencv
    cv::Mat o_u1_1(inputHeight, inputWidth, CV_8U);
    cv::Mat o_u1_2(inputHeight, inputWidth, CV_8U);
    cv::Mat o_f1_1(inputHeight, inputWidth, CV_32F);
    cv::Mat o_f1_2(inputHeight, inputWidth, CV_32F);
    cv::Mat o_u3_1(inputHeight, inputWidth, CV_8UC3);
    cv::Mat o_u3_2(inputHeight, inputWidth, CV_8UC3);
    cv::Mat o_f3_1(inputHeight, inputWidth, CV_32FC3);
    cv::Mat o_f3_2(inputHeight, inputWidth, CV_32FC3);
    cv::Mat o_u4_1(inputHeight, inputWidth, CV_8UC4);
    cv::Mat o_u4_2(inputHeight, inputWidth, CV_8UC4);
    cv::Mat o_f4_1(inputHeight, inputWidth, CV_32FC4);
    cv::Mat o_f4_2(inputHeight, inputWidth, CV_32FC4);
    //fastcv
    Mat<uchar, 1> u1_1(inputHeight, inputWidth);
    Mat<uchar, 1> u1_2(inputHeight, inputWidth);
    Mat<float, 1> f1_1(inputHeight, inputWidth);
    Mat<float, 1> f1_2(inputHeight, inputWidth);
    Mat<uchar, 3> u3_1(inputHeight, inputWidth);
    Mat<uchar, 3> u3_2(inputHeight, inputWidth);
    Mat<float, 3> f3_1(inputHeight, inputWidth);
    Mat<float, 3> f3_2(inputHeight, inputWidth);
    Mat<uchar, 4> u4_1(inputHeight, inputWidth);
    Mat<uchar, 4> u4_2(inputHeight, inputWidth);
    Mat<float, 4> f4_1(inputHeight, inputWidth);
    Mat<float, 4> f4_2(inputHeight, inputWidth);

    //for BGR2NV21
    Mat<uchar, 1> NV21(inputHeight/2*3, inputWidth);
    Mat<uchar, 1> NV12(inputHeight/2*3, inputWidth);

    refrensh();

#endif

#if defined(USE_CUDA)
    cv::gpu::GpuMat o_cuda_u1_1; o_cuda_u1_1.upload(o_u1_1);
    cv::gpu::GpuMat o_cuda_u1_2; o_cuda_u1_1.upload(o_u1_2);
    cv::gpu::GpuMat o_cuda_f1_1; o_cuda_f1_1.upload(o_f1_1);
    cv::gpu::GpuMat o_cuda_f1_2; o_cuda_f1_1.upload(o_f1_2);
    cv::gpu::GpuMat o_cuda_u3_1; o_cuda_u3_1.upload(o_u3_1);
    cv::gpu::GpuMat o_cuda_u3_2; o_cuda_u3_1.upload(o_u3_2);
    cv::gpu::GpuMat o_cuda_f3_1; o_cuda_f3_1.upload(o_f3_1);
    cv::gpu::GpuMat o_cuda_f3_2; o_cuda_f3_1.upload(o_f3_2);
#endif


    //warm up
    cvtColor<uchar, 3, uchar, 1, BGR2NV21>(u3_1, &NV21);
    //BGR2NV21
    refrensh();
    show(bgr2nv21_u_3);
    show_(inputWidth);
    gettime((cvtColor<uchar, 3, uchar, 1, BGR2NV21>(u3_1, &NV21)));
    show(bgr2nv21_u_4)
    show_(inputWidth);
    gettime((cvtColor<uchar, 4, uchar, 1, BGR2NV21>(u4_1, &NV21)));

    //NV212BGR
    show(nv212bgr_u_3)
    show_(inputWidth);
    gettime((cvtColor<uchar, 1, uchar, 3, NV212BGR>(NV21, &u3_1)));
    show(nv212bgr_u_4)
    show_(inputWidth);
    gettime((cvtColor<uchar, 1, uchar, 4, NV212BGR>(NV21, &u4_1)));
    
    //RGB2NV21
    refrensh();
    show(rgb2nv21_u_3);
    show_(inputWidth);
    gettime((cvtColor<uchar, 3, uchar, 1, RGB2NV21>(u3_1, &NV21)));
    show(rgb2nv21_u_4)
    show_(inputWidth);
    gettime((cvtColor<uchar, 4, uchar, 1, RGB2NV21>(u4_1, &NV21)));

    //NV212BGR
    show(nv212rgb_u_3)
    show_(inputWidth);
    gettime((cvtColor<uchar, 1, uchar, 3, NV212RGB>(NV21, &u3_1)));
    show(nv212rgb_u_4)
    show_(inputWidth);
    gettime((cvtColor<uchar, 1, uchar, 4, NV212RGB>(NV21, &u4_1)));

    //BGR2NV12
    refrensh();
    show(bgr2nv12_u_3);
    show_(inputWidth);
    gettime((cvtColor<uchar, 3, uchar, 1, BGR2NV12>(u3_1, &NV12)));
    show(bgr2nv12_u_4)
    show_(inputWidth);
    gettime((cvtColor<uchar, 4, uchar, 1, BGR2NV12>(u4_1, &NV12)));

    //NV122BGR
    show(nv122bgr_u_3)
    show_(inputWidth);
    gettime((cvtColor<uchar, 1, uchar, 3, NV122BGR>(NV12, &u3_1)));
    show(nv122bgr_u_4)
    show_(inputWidth);
    gettime((cvtColor<uchar, 1, uchar, 4, NV122BGR>(NV12, &u4_1)));
    
    //GRAY2BGR(A)
    refrensh();
    show(gray2bgr_u_3);
    show_(inputWidth);
    gettime((cvtColor<uchar, 1, uchar, 3, GRAY2BGR>(u1_1, &u3_1)));
    gettime_opencv((cv::cvtColor(o_u1_1, o_u3_1, CV_GRAY2BGR)));
    show(gray2bgr_u_4)
    show_(inputWidth);
    gettime((cvtColor<uchar, 1, uchar, 4, GRAY2BGR>(u1_1, &u4_1)));
    gettime_opencv((cv::cvtColor(o_u1_1, o_u4_1, CV_GRAY2BGRA)));

    show(bgr2gray_u_3);
    show_(inputWidth);
    gettime((cvtColor<uchar, 3, uchar, 1, BGR2GRAY>(u3_1, &u1_1)));
    gettime_opencv((cv::cvtColor(o_u3_1, o_u1_1, CV_BGR2GRAY)));
    show(bgr2gray_u_4)
    show_(inputWidth);
    gettime((cvtColor<uchar, 4, uchar, 1, BGR2GRAY>(u4_1, &u1_1)));
    gettime_opencv((cv::cvtColor(o_u4_1, o_u1_1, CV_BGRA2GRAY)));

    show(gray2bgr_f_3);
    show_(inputWidth);
    gettime((cvtColor<float, 1, float, 3, GRAY2BGR>(f1_1, &f3_1)));
    gettime_opencv((cv::cvtColor(o_f1_1, o_f3_1, CV_GRAY2BGR)));
    show(gray2bgr_f_4)
    show_(inputWidth);
    gettime((cvtColor<float, 1, float, 4, GRAY2BGR>(f1_1, &f4_1)));
    gettime_opencv((cv::cvtColor(o_f1_1, o_f4_1, CV_GRAY2BGRA)));

    show(bgr2gray_f_3);
    show_(inputWidth);
    gettime((cvtColor<float, 3, float, 1, BGR2GRAY>(f3_1, &f1_1)));
    gettime_opencv((cv::cvtColor(o_f3_1, o_f1_1, CV_BGR2GRAY)));
    show(bgr2gray_f_4)
    show_(inputWidth);
    gettime((cvtColor<float, 4, float, 1, BGR2GRAY>(f4_1, &f1_1)));
    gettime_opencv((cv::cvtColor(o_f4_1, o_f1_1, CV_BGRA2GRAY)));

    //BGR2BGRA
    refrensh();
    show(bgr2bgra_u_4);
    show_(inputWidth);
    gettime((cvtColor<uchar, 3, uchar, 4, BGR2BGRA>(u3_1, &u4_1)));
    gettime_opencv((cv::cvtColor(o_u3_1, o_u4_1, CV_BGR2BGRA)));
    show(bgra2bgr_u_4)
    show_(inputWidth);
    gettime((cvtColor<uchar, 4, uchar, 3, BGRA2BGR>(u4_1, &u3_1)));
    gettime_opencv((cv::cvtColor(o_u4_1, o_u3_1, CV_BGRA2BGR)));

    show(bgr2bgra_f_4);
    show_(inputWidth);
    gettime((cvtColor<float, 3, float, 4, BGR2BGRA>(f3_1, &f4_1)));
    gettime_opencv((cv::cvtColor(o_f3_1, o_f4_1, CV_BGR2BGRA)));
    show(bgra2bgr_f_4)
    show_(inputWidth);
    gettime((cvtColor<float, 4, float, 3, BGRA2BGR>(f4_1, &f3_1)));
    gettime_opencv((cv::cvtColor(o_f4_1, o_f3_1, CV_BGRA2BGR)));

    //BGR2RGB
    refrensh();
    show(bgr2rgb_u_3);
    show_(inputWidth);
    gettime((cvtColor<uchar, 3, uchar, 3, BGR2RGB>(u3_1, &u3_2)));
    gettime_opencv((cv::cvtColor(o_u3_1, o_u3_2, CV_BGR2RGB)));
    show(bgr2rgb_u_4)
    show_(inputWidth);
    gettime((cvtColor<uchar, 4, uchar, 4, BGR2RGB>(u4_1, &u4_2)));
    //ettime_opencv((cv::cvtColor(o_u4_2, o_u4_2, CV_BGR2RGB)));

    show(bgr2rgb_f_3);
    show_(inputWidth);
    gettime((cvtColor<float, 3, float, 3, BGR2RGB>(f3_1, &f3_2)));
    gettime_opencv((cv::cvtColor(o_f3_1, o_f3_2, CV_BGR2RGB)));
    show(bgr2rgb_f_4)
    show_(inputWidth);
    gettime((cvtColor<float, 4, float, 4, BGR2RGB>(f4_1, &f4_2)));
   // gettime_opencv((cv::cvtColor(o_f4_1, o_f4_2, CV_BGR2RGB)));


    //BGR LAB
    refrensh();
    show(bgr2lab_u_3);
    show_(inputWidth);
    gettime((cvtColor<uchar, 3, uchar, 3, BGR2LAB>(u3_1, &u3_2)));
    gettime_opencv((cv::cvtColor(o_u3_1, o_u3_2, CV_BGR2Lab)));
    show(lab2bgr_u_3)
    show_(inputWidth);
    gettime((cvtColor<uchar, 3, uchar, 3, LAB2BGR>(u3_2, &u3_1)));
    gettime_opencv((cv::cvtColor(o_u3_2, o_u3_1, CV_Lab2BGR)));

    show(bgr2lab_f_3);
    show_(inputWidth);
    gettime((cvtColor<float, 3, float, 3, BGR2LAB>(f3_1, &f3_2)));
    gettime_opencv((cv::cvtColor(o_f3_1, o_f3_2, CV_BGR2Lab)));
    show(lab2bgr_f_3)
    show_(inputWidth);
    gettime((cvtColor<float, 3, float, 3, LAB2BGR>(f3_2, &f3_1)));
    gettime_opencv((cv::cvtColor(o_f3_2, o_f3_1, CV_Lab2BGR)));

    //BGR YCrYb
    refrensh();
    show(bgr2ycrcb_u_3);
    show_(inputWidth);
    gettime((cvtColor<uchar, 3, uchar, 3, BGR2YCrCb>(u3_1, &u3_2)));
    gettime_opencv((cv::cvtColor(o_u3_1, o_u3_2, CV_BGR2YCrCb)));
    show(ycrcb2bgr_u_3)
    show_(inputWidth);
    gettime((cvtColor<uchar, 3, uchar, 3, YCrCb2BGR>(u3_2, &u3_1)));
    gettime_opencv((cv::cvtColor(o_u3_2, o_u3_1, CV_YCrCb2BGR)));

    show(bgr2ycrcb_f_3);
    show_(inputWidth);
    gettime((cvtColor<float, 3, float, 3, BGR2YCrCb>(f3_1, &f3_2)));
    gettime_opencv((cv::cvtColor(o_f3_1, o_f3_2, CV_BGR2YCrCb)));
    show(ycrcb2bgr_u_3)
    show_(inputWidth);
    gettime((cvtColor<float, 3, float, 3, YCrCb2BGR>(f3_2, &f3_1)));
    gettime_opencv((cv::cvtColor(o_f3_2, o_f3_1, CV_YCrCb2BGR)));

     //BGR HSV
    refrensh();
    show(bgr2hsv_u_3);
    show_(inputWidth);
    gettime((cvtColor<uchar, 3, uchar, 3, BGR2HSV>(u3_1, &u3_2)));
    gettime_opencv((cv::cvtColor(o_u3_1, o_u3_2, CV_BGR2HSV)));
    show(hsv2bgr_u_3)
    show_(inputWidth);
    gettime((cvtColor<uchar, 3, uchar, 3, HSV2BGR>(u3_2, &u3_1)));
    gettime_opencv((cv::cvtColor(o_u3_2, o_u3_1, CV_HSV2BGR)));

    show(bgr2hsv_f_3);
    show_(inputWidth);
    gettime((cvtColor<float, 3, float, 3, BGR2HSV>(f3_1, &f3_2)));
    gettime_opencv((cv::cvtColor(o_f3_1, o_f3_2, CV_BGR2HSV)));
    show(hsv2bgr_f_3)
    show_(inputWidth);
    gettime((cvtColor<float, 3, float, 3, HSV2BGR>(f3_2, &f3_1)));
    gettime_opencv((cv::cvtColor(o_f3_2, o_f3_1, CV_HSV2BGR)));


    
}

int main(){
    benchmark(1280, 720);
    benchmark(1920, 1080);
    benchmark(4200, 3136);
}
