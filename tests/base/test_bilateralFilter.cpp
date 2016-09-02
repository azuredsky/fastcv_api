//#include <fastcv.hpp>
//#include <gtest/gtest.h>
//#include "opencv2/opencv.hpp"
//#include <sys/time.h>
//#include <opencv2/gpu/gpu.hpp>
//#include<opencv2/ocl/ocl.hpp>
#include "test_head.hpp"
using namespace HPC::fastcv;

#define N 1000
//#define PRINT_ALL

struct imageParam{
    size_t height;
    size_t width;
    int diameter;
    float color;
    float space;
    int spadding;
    int dpadding;
};


class BilateralFilterTest : public testing::Test
{
    public:
        virtual void SetUp()
        {
           // std::vector<imageParam>::iterator iter = imageparams.begin();
            if(globalEnvironment::Instance()->testconfig.hasSection("BilateralFilterTest"))
            {
                Cfg::cfg_type vec_arg = globalEnvironment::Instance()->testconfig.sectionConfigVec("BilateralFilterTest");
                int num = vec_arg.size();
                for(int i = 0;i < num; i++)
                {
                    ASSERT_TRUE(vec_arg[i].find("height")!=vec_arg[i].end())<<"argument:height not found";
                    ASSERT_TRUE(vec_arg[i].find("width")!=vec_arg[i].end())<<"argument:width not found";
                    ASSERT_TRUE(vec_arg[i].find("diameter")!=vec_arg[i].end())<<"argument:diameter not found";
                    ASSERT_TRUE(vec_arg[i].find("color")!=vec_arg[i].end())<<"argument:color not found";
                    ASSERT_TRUE(vec_arg[i].find("space")!=vec_arg[i].end())<<"argument:space not found";
                    ASSERT_TRUE(vec_arg[i].find("spadding")!=vec_arg[i].end())<<"argument:spadding not found";
                    ASSERT_TRUE(vec_arg[i].find("dpadding")!=vec_arg[i].end())<<"argument:dpadding not found";
                    imagep.height = std::stoul(vec_arg[i]["height"]);
                    imagep.width = std::stoul(vec_arg[i]["width"]);
                    imagep.diameter = std::stoi(vec_arg[i]["diameter"]);
                    imagep.color = std::stof(vec_arg[i]["color"]);
                    imagep.space = std::stof(vec_arg[i]["space"]);
                    imagep.spadding = std::stoi(vec_arg[i]["spadding"]);
                    imagep.dpadding = std::stoi(vec_arg[i]["dpadding"]);
                   // iter = imageparams.insert(iter,imagep);


                }
            }
            else
            {
                imagep = {5,5,3,0.5,0.5,7,21};
                //imageparams.insert(iter,imagep);
            }

        }
        virtual void TearDown()
        {
           // if(!imageparams.empty())
            //    imageparams.clear();
        }
    public:
        imageParam imagep;
       // std::vector<imageParam> imageparams;
};

template<typename T, int nc>
void checkResult(const T *data1, const T *data2, const int height, const int width, int dstep, const float diff_THR) {
    double max = DBL_MIN;
    double min = DBL_MAX;
    double temp_diff = 0.0;

    int num = height*width *nc;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width*nc; j++){
            float val1 = data1[i * dstep +j];
            float val2 = data2[i * dstep + j];

            temp_diff = fabs(val1 - val2);

#ifdef PRINT_ALL
            std::cout << "Print data12: " << i << " : " <<  j << " : "<< (float) val1 << " : " << (float) val2 << " : " << temp_diff << std::endl;
#endif
            max = (temp_diff > max) ? temp_diff : max;
            min = (temp_diff < min) ? temp_diff : min;
        }
    }
    EXPECT_LT(max, diff_THR);
}


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

template<typename T, int val>
void randomPaddingData(T *data, int height, int width, int widthStride){
        size_t tmp;
        clock_t ct = clock();
        srand((unsigned int)ct);

        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++)
            {
                tmp = rand()% 255;
                data[i*widthStride + j] = (T) ( (float)tmp/(float)val );
                //printf("%d\n",data[i]);
            }
        }
}


template<typename T, int ncSrc, int ncDst, int nc, BorderType bt, int val_div>
int testBilateralFilterFunc(imageParam p, float diff_THR) {
    size_t input_width = p.width;
    size_t input_height = p.height;
    size_t input_channels = ncSrc;
    int diameter = p.diameter;
    double color = p.color;
    double space = p.space;
    size_t output_width = p.width;
    size_t output_height = p.height;
    size_t output_channels = ncDst;
    int spadding = p.spadding;
    int dpadding = p.dpadding;
    struct timeval start, end;

    printf("Input Mat scale: width*height*channels = %ld*%ld*%ld\n", input_width, input_height, input_channels);

    size_t step = (input_width * input_channels + spadding)*sizeof(T);
    T* input_data = (T*)malloc(input_height*step);
    randomPaddingData<T, val_div>(input_data, input_height,
            input_width*input_channels, input_width * input_channels + spadding);

#if defined(FASTCV_USE_CUDA) 
    Mat<T, ncSrc> src(input_height, input_width, step);
    src.fromHost(input_data);

    const Mat<T, ncSrc,EcoEnv_t::ECO_ENV_X86> src_x86(input_height, input_width, input_data, step);
#elif defined(FASTCV_USE_OCL)
    Mat<T, ncSrc> src(input_height, input_width, step);
    src.fromHost(input_data);
#else
    const Mat<T, ncSrc> src(input_height, input_width, input_data, step);
#endif

    size_t dstep = (output_width * output_channels + dpadding)*sizeof(T);
    T* output_data = (T*)malloc(output_height*dstep);
    T* output_data_opencv = (T*)malloc(output_height*dstep);
    randomPaddingData<T, val_div>(output_data, output_height,
            output_width*output_channels, output_width * output_channels + dpadding);

    randomPaddingData<T, val_div>(output_data_opencv, output_height,
            output_width*output_channels, output_width * output_channels + dpadding);

#if defined(FASTCV_USE_CUDA)

    Mat<T, ncDst> dst(input_height, input_width, dstep);
    dst.fromHost(output_data);

    Mat<T, ncDst,EcoEnv_t::ECO_ENV_X86> dst_x86(output_height, output_width, output_data, dstep);


    //warm up
    HPCStatus_t sta;// = bilateralFilter<T, ncSrc, ncDst, nc, bt>(src, diameter, color, space, &dst);
    HPCStatus_t sta_x86;


//cuda
    gettimeofday(&start, NULL);
    sta = bilateralFilter<bt, T, ncSrc, ncDst, nc>(src, diameter, color, space, &dst);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    double time_fastcv_cuda = timingExec(start, end);
    EXPECT_EQ(HPC_SUCCESS, sta);

    //x86
    gettimeofday(&start,NULL);
    for(int i = 0; i < 10; i++){
        sta_x86 = bilateralFilter<bt, T, ncSrc, ncDst, nc>(src_x86, diameter, color, space, &dst_x86);
    }

    gettimeofday(&end, NULL);
    double time_fastcv_x86 = timingExec(start, end);
    EXPECT_EQ(HPC_SUCCESS, sta_x86);

    //call opencv
    int opencv_type = 0;
    if(nc == 1){
        opencv_type = (sizeof(T) == sizeof(float))? CV_32FC1 :CV_8UC1;
    }
    if(nc == 3){
        opencv_type = (sizeof(T) == sizeof(float))? CV_32FC3 :CV_8UC3;
    }

    cv::Mat src_opencv(input_height, input_width, opencv_type, input_data, step);
    cv::Mat dst_opencv(input_height, input_width, opencv_type, output_data_opencv, dstep);


    //warm up cuda
    cv::bilateralFilter(src_opencv, dst_opencv, diameter, color, space);

    gettimeofday(&start, NULL);
    cv::bilateralFilter(src_opencv, dst_opencv, diameter, color, space);
    gettimeofday(&end, NULL);
    double time_opencv_in_cuda = timingExec(start, end);


    //warm up  x86
     gettimeofday(&start, NULL);
    for(int i = 0; i < 10; i++);
        //cv::bilateralFilter(src_opencv, dst_opencv, diameter, color, space);
    gettimeofday(&end, NULL);
    double time_opencv_in_x86 = timingExec(start, end);



    printf("fastcv_cuda cost %f  opencv_in_cuda cost %f\n", time_fastcv_cuda, time_opencv_in_cuda);
    printf("fastcv_x86 cost %f  opencv_in_x86 cost %f\n", time_fastcv_x86, time_opencv_in_x86);

#elif defined (FASTCV_USE_OCL)
    Mat<T, ncDst> dst(input_height, input_width, dstep);
    dst.fromHost(output_data);

    //warm up
    HPCStatus_t sta = bilateralFilter<bt, T, ncSrc, ncDst, nc>(src, diameter, color, space, &dst);
    gettimeofday(&start, NULL);
    for(int i=0;i<N;i++){
        sta = bilateralFilter<bt, T, ncSrc, ncDst, nc>(src, diameter, color, space, &dst);
        clFinish(opencl.getCommandQueue());
    }
    gettimeofday(&end, NULL);
    double time_fastcv = timingExec(start, end);
    EXPECT_EQ(HPC_SUCCESS, sta);

    //call opencv
    int opencv_type = 0;
    if(nc == 1){
        opencv_type = (sizeof(T) == sizeof(float))? CV_32FC1 :CV_8UC1;
    }
    if(nc == 3){
        opencv_type = (sizeof(T) == sizeof(float))? CV_32FC3 :CV_8UC3;
    }
    cv::Mat src_opencv(input_height, input_width, opencv_type, input_data, step);
    cv::Mat dst_opencv(input_height, input_width, opencv_type, output_data, dstep);
  //  memcpy(src_opencv.ptr<T>(), input_data, input_data_num*sizeof(T));
    cv::ocl::oclMat src_opencvD(src_opencv);
    cv::ocl::oclMat dst_opencvD;

    //warm up
    cv::ocl::bilateralFilter(src_opencvD, dst_opencvD, diameter, color, space);

    gettimeofday(&start, NULL);
    for(int i=0;i<N;i++){
        cv::ocl::bilateralFilter(src_opencvD, dst_opencvD, diameter, color, space);
        cv::ocl::finish();
    }
    gettimeofday(&end, NULL);
    double time_opencv = timingExec(start, end);
    dst_opencvD.download(dst_opencv);
    printf("fastcv cost %f opencv cost %f\n", time_fastcv, time_opencv);


#else
    Mat<T, ncDst> dst(output_height, output_width, output_data, dstep);
   /* for(size_t i=0; i<input_data_num; i++){
        input_data[i] = (T)(-1);
    }
    dst.fromHost(input_data);*/

    //warm up
    HPCStatus_t sta;// = bilateralFilter<T, ncSrc, ncDst, nc, bt>(src, diameter, color, space, &dst);

    gettimeofday(&start, NULL);
    for(int i = 0; i < 10; i++){
        sta = bilateralFilter<bt, T, ncSrc, ncDst, nc>(src, diameter, color, space, &dst);
    }

    gettimeofday(&end, NULL);
    double time_fastcv = timingExec(start, end);
    EXPECT_EQ(HPC_SUCCESS, sta);

    //call opencv
    int opencv_type = 0;
    if(nc == 1){
        opencv_type = (sizeof(T) == sizeof(float))? CV_32FC1 :CV_8UC1;
    }
    if(nc == 3){
        opencv_type = (sizeof(T) == sizeof(float))? CV_32FC3 :CV_8UC3;
    }

    cv::Mat src_opencv(input_height, input_width, opencv_type, input_data, step);
    cv::Mat dst_opencv(input_height, input_width, opencv_type, output_data_opencv, dstep);

   // memcpy(src_opencv.ptr<T>(), input_data, input_data_num*sizeof(T));

    //warm up
    cv::bilateralFilter(src_opencv, dst_opencv, diameter, color, space);

    gettimeofday(&start, NULL);
    for(int i = 0; i < 10; i++);
        //cv::bilateralFilter(src_opencv, dst_opencv, diameter, color, space);
    gettimeofday(&end, NULL);
    double time_opencv = timingExec(start, end);

    printf("fastcv cost %f opencv cost %f\n", time_fastcv, time_opencv);

#endif

    // check results
#if defined(FASTCV_USE_CUDA)
   // size_t output_data_num = output_width*output_height*output_channels;
   // T* dst_data_x86 = (T*)malloc(output_data_num*sizeof(T));
    memset(output_data,0,output_height * dstep);
    dst.toHost(output_data);
    checkResult<T, ncDst>(output_data, dst_opencv.ptr<T>(), output_height, output_width, dstep/sizeof(T), diff_THR);

    checkResult<T, ncDst>((T*)dst_x86.ptr(), dst_opencv.ptr<T>(), output_height,output_width, dstep/sizeof(T), diff_THR);
    //free(dst_data_x86);   
#elif defined(FASTCV_USE_OCL)
    memset(output_data,0,output_height * dstep);
    dst.toHost(output_data);
    checkResult<T, ncDst>(output_data, dst_opencv.ptr<T>(), output_height, output_width, dstep/sizeof(T), diff_THR);
#else
    checkResult<T, ncDst>((T*)dst.ptr(), dst_opencv.ptr<T>(), output_height,
            output_width, dstep/sizeof(T), diff_THR);
#endif

    free(input_data);
    free(output_data);
    free(output_data_opencv);
    return 0;
}


TEST_F(BilateralFilterTest, float){
    imageParam p;
  //  ASSERT_TRUE(!imageparams.empty());

  //  for(int i = 0;i<imageparams.size();i++)
  //  {
        p = imagep;
        testBilateralFilterFunc<uchar, 1, 1, 1, BORDER_DEFAULT, 1>(p, 1.01);
        testBilateralFilterFunc<uchar, 3, 3, 3, BORDER_DEFAULT, 1>(p, 1.01);


}

#undef PRINT_ALL
#undef N




