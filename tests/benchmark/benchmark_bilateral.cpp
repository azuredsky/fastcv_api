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


void benchmark(int inputWidth, int inputHeight, int diameter){
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
    cvtColor<uchar, 3, uchar, 1, BGR2GRAY>(u3_1, &u1_1);

    //BGR2BGRA
    refrensh();
    if(diameter == 3){
        show(bilateral-d-3_u_1);
    }
    if(diameter == 5){
        show(bilateral-d-5_u_1);
    }
    if(diameter == 7){
        show(bilateral-d-7_u_1);
    }
    show_(inputWidth);
    gettime((bilateralFilter<uchar, 1, 1, 1, BORDER_DEFAULT>(u1_1, diameter, 0.5, 0.5, &u1_2)));
    gettime_opencv((cv::bilateralFilter(o_u1_1, o_u1_2, diameter, 0.5, 0.5)));


    if(diameter == 3){
        show(bilateral-d-3_u_3);
    }
    if(diameter == 5){
        show(bilateral-d-5_u_3);
    }
    if(diameter == 7){
        show(bilateral-d-7_u_3);
    }
    show_(inputWidth);
    gettime((bilateralFilter<uchar, 3, 3, 3, BORDER_DEFAULT>(u3_1, diameter, 0.5, 0.5, &u3_2)));
    gettime_opencv((cv::bilateralFilter(o_u3_1, o_u3_2, diameter, 0.5, 0.5)));


    if(diameter == 3){
        show(bilateral-d-3_f_1);
    }
    if(diameter == 5){
        show(bilateral-d-5_f_1);
    }
    if(diameter == 7){
        show(bilateral-d-7_f_1);
    }
    show_(inputWidth);
    gettime((bilateralFilter<float, 1, 1, 1, BORDER_DEFAULT>(f1_1, diameter, 0.5, 0.5, &f1_2)));
    gettime_opencv((cv::bilateralFilter(o_f1_1, o_f1_2, diameter, 0.5, 0.5)));


    if(diameter == 3){
        show(bilateral-d-3_f_3);
    }
    if(diameter == 5){
        show(bilateral-d-5_f_3);
    }
    if(diameter == 7){
        show(bilateral-d-7_f_3);
    }
    show_(inputWidth);
    gettime((bilateralFilter<float, 3, 3, 3, BORDER_DEFAULT>(f3_1, diameter, 0.5, 0.5, &f3_2)));
    gettime_opencv((cv::bilateralFilter(o_f3_1, o_f3_2, diameter, 0.5, 0.5)));



    
}

int main(){
    benchmark(640, 480, 3);
    benchmark(1280, 720, 3);
    benchmark(1920, 1080, 3);
    benchmark(640, 480, 5);
    benchmark(1280, 720, 5);
    benchmark(1920, 1080, 5);
    benchmark(640, 480, 7);
    benchmark(1280, 720, 7);
    benchmark(1920, 1080, 7);
}
