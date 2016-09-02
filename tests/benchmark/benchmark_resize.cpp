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


template<InterpolationType type>
void benchmark(int inputWidth, int inputHeight, int outputWidth, int outputHeight){
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

    //dst
    //opencv
    cv::Mat o_u1_d1(outputHeight, outputWidth, CV_8U);
    cv::Mat o_u1_d2(outputHeight, outputWidth, CV_8U);
    cv::Mat o_f1_d1(outputHeight, outputWidth, CV_32F);
    cv::Mat o_f1_d2(outputHeight, outputWidth, CV_32F);
    cv::Mat o_u3_d1(outputHeight, outputWidth, CV_8UC3);
    cv::Mat o_u3_d2(outputHeight, outputWidth, CV_8UC3);
    cv::Mat o_f3_d1(outputHeight, outputWidth, CV_32FC3);
    cv::Mat o_f3_d2(outputHeight, outputWidth, CV_32FC3);
    cv::Mat o_u4_d1(outputHeight, outputWidth, CV_8UC4);
    cv::Mat o_u4_d2(outputHeight, outputWidth, CV_8UC4);
    cv::Mat o_f4_d1(outputHeight, outputWidth, CV_32FC4);
    cv::Mat o_f4_d2(outputHeight, outputWidth, CV_32FC4);
    //fastcv
    Mat<uchar, 1> u1_d1(outputHeight, outputWidth);
    Mat<uchar, 1> u1_d2(outputHeight, outputWidth);
    Mat<float, 1> f1_d1(outputHeight, outputWidth);
    Mat<float, 1> f1_d2(outputHeight, outputWidth);
    Mat<uchar, 3> u3_d1(outputHeight, outputWidth);
    Mat<uchar, 3> u3_d2(outputHeight, outputWidth);
    Mat<float, 3> f3_d1(outputHeight, outputWidth);
    Mat<float, 3> f3_d2(outputHeight, outputWidth);
    Mat<uchar, 4> u4_d1(outputHeight, outputWidth);
    Mat<uchar, 4> u4_d2(outputHeight, outputWidth);
    Mat<float, 4> f4_d1(outputHeight, outputWidth);
    Mat<float, 4> f4_d2(outputHeight, outputWidth);

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



    //resize
    refrensh();
    if(INTERPOLATION_TYPE_LINEAR == type){
        show(resize-linear_u_1);
    }
    if(INTERPOLATION_TYPE_NEAREST_POINT == type){
        show(resize-nearest_u_1);
    }
    show_(inputWidth);
    gettime((resize<uchar, 1, uchar, 1, 1, type>(u1_1, &u1_d1)));
    gettime_opencv((cv::resize(o_u1_1, o_u1_d1, o_u1_d1.size(), 0, 0, type)));

    if(INTERPOLATION_TYPE_LINEAR == type){
        show(resize-linear_u_3);
    }
    if(INTERPOLATION_TYPE_NEAREST_POINT == type){
        show(resize-nearest_u_3);
    }
    show_(inputWidth);
    gettime((resize<uchar, 3, uchar, 3, 3, type>(u3_1, &u3_d1)));
    gettime_opencv((cv::resize(o_u3_1, o_u3_d1, o_u3_d1.size(), 0, 0, type)));

    if(INTERPOLATION_TYPE_LINEAR == type){
        show(resize-linear_u_4);
    }
    if(INTERPOLATION_TYPE_NEAREST_POINT == type){
        show(resize-nearest_u_4);
    }
    show_(inputWidth);
    gettime((resize<uchar, 4, uchar, 4, 4, type>(u4_1, &u4_d1)));
    gettime_opencv((cv::resize(o_u4_1, o_u4_d1, o_u4_d1.size(), 0, 0, type)));

    if(INTERPOLATION_TYPE_LINEAR == type){
        show(resize-linear_f_1);
    }
    if(INTERPOLATION_TYPE_NEAREST_POINT == type){
        show(resize-nearest_f_1);
    }
    show_(inputWidth);
    gettime((resize<float, 1, float, 1, 1, type>(f1_1, &f1_d1)));
    gettime_opencv((cv::resize(o_f1_1, o_f1_d1, o_f1_d1.size(), 0, 0, type)));

    if(INTERPOLATION_TYPE_LINEAR == type){
        show(resize-linear_f_3);
    }
    if(INTERPOLATION_TYPE_NEAREST_POINT == type){
        show(resize-nearest_f_3);
    }
    show_(inputWidth);
    gettime((resize<float, 3, float, 3, 3, type>(f3_1, &f3_d1)));
    gettime_opencv((cv::resize(o_f3_1, o_f3_d1, o_f3_d1.size(), 0, 0, type)));

    if(INTERPOLATION_TYPE_LINEAR == type){
        show(resize-linear_f_4);
    }
    if(INTERPOLATION_TYPE_NEAREST_POINT == type){
        show(resize-nearest_f_4);
    }
    show_(inputWidth);
    gettime((resize<float, 4, float, 4, 4, type>(f4_1, &f4_d1)));
    gettime_opencv((cv::resize(o_f4_1, o_f4_d1, o_f4_d1.size(), 0, 0, type)));

    
}

int main(){
    benchmark<INTERPOLATION_TYPE_LINEAR>(640, 480, 1920, 1080);
    benchmark<INTERPOLATION_TYPE_LINEAR>(1920, 1080, 640, 480);
    benchmark<INTERPOLATION_TYPE_NEAREST_POINT>(640, 480, 1920, 1080);
    benchmark<INTERPOLATION_TYPE_NEAREST_POINT>(1920, 1080, 640, 480);
}
