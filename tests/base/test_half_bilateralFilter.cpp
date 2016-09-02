#include "test_head.hpp"
#include <opencv2/gpu/gpu.hpp>
#include<opencv2/ocl/ocl.hpp>
using namespace HPC::fastcv;
using namespace uni::half;

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



class halfBilateralFilterTest : public testing::Test
{
    public:
        virtual void SetUp()
        {
            std::vector<imageParam>::iterator iter = imageparams.begin();
            if(globalEnvironment::Instance()->testconfig.hasSection("halfBilateralFilterTest"))
            {
                Cfg::cfg_type vec_arg = globalEnvironment::Instance()->testconfig.sectionConfigVec("halfBilateralFilterTest");
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
                   iter = imageparams.insert(iter,imagep);


                }
            }
            else
            {
                imagep = {5,5,3,0.5,0.5,7,11};
                imageparams.insert(iter,imagep);
            }

        }
        virtual void TearDown()
        {
            if(!imageparams.empty())
                imageparams.clear();
        }
    public:
        imageParam imagep;
        std::vector<imageParam> imageparams;
};


template<typename T, int nc>
void checkResult(const T *data1, const float *data2, const int height, const int width, int dstep, const float diff_THR) {
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

    int inStride =  (input_width * input_channels + spadding);
    size_t step = inStride *sizeof(T);
    size_t step_opencv = inStride*sizeof(float);
    T* input_data = (T*)malloc(input_height*step);
    float* input_data_opencv = (float*)malloc(input_height*step_opencv);
    randomPaddingData<T, val_div>(input_data, input_height,
            input_width*input_channels, input_width * input_channels + spadding);

    for(int i = 0;i<input_height * inStride ;i++){
        input_data_opencv[i] = input_data[i];
    }
#if defined(FASTCV_USE_CUDA)
    Mat<T, ncSrc> src(input_height, input_width, step);
    src.fromHost(input_data);

 //   const Mat<T, ncSrc,EcoEnv_t::ECO_ENV_X86> src_x86(input_height, input_width, input_data, step);

#elif defined(FASTCV_USE_OCL)
    Mat<T, ncSrc> src(input_height, input_width, step);
    src.fromHost(input_data);
#else
    const Mat<T, ncSrc> src(input_height, input_width, input_data, step);
#endif

    int outStride =   (output_width * output_channels + dpadding);
    size_t dstep = outStride * sizeof(T);
    size_t dstep_opencv = outStride * sizeof(float);
    T* output_data = (T*)malloc(output_height*dstep);
    float* output_data_opencv = (float*)malloc(output_height*dstep_opencv);

#if defined(FASTCV_USE_CUDA)

    Mat<T, ncDst> dst(input_height, input_width, dstep);
    dst.fromHost(output_data);

    //warm up
    HPCStatus_t sta;

    gettimeofday(&start, NULL);
    sta = bilateralFilter<bt, T, ncSrc, ncDst, nc>(src, diameter, color, space, &dst);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    double time_fastcv = timingExec(start, end);
    EXPECT_EQ(HPC_SUCCESS, sta);

    //call opencv
    int opencv_type = 0;
    if(nc == 1){
        opencv_type = (sizeof(T) == sizeof(uchar))? CV_8UC1  :CV_32FC1;
    }
    if(nc == 3){
        opencv_type = (sizeof(T) == sizeof(uchar))? CV_8UC3  :CV_32FC3;
    }

    cv::Mat src_opencv(input_height, input_width, opencv_type, input_data_opencv, step_opencv);
    cv::Mat dst_opencv(input_height, input_width, opencv_type, output_data_opencv, dstep_opencv);



    //warm up
    cv::bilateralFilter(src_opencv, dst_opencv, diameter, color, space);

    gettimeofday(&start, NULL);
    cv::bilateralFilter(src_opencv, dst_opencv, diameter, color, space);
    gettimeofday(&end, NULL);
    double time_opencv = timingExec(start, end);

    printf("fastcv cost %f  opencv cost %f\n", time_fastcv, time_opencv);

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

#endif

    // check results
#if defined(FASTCV_USE_CUDA) || defined(FASTCV_USE_OCL)
    memset(output_data,0,output_height * dstep);
    dst.toHost(output_data);
    checkResult<T, ncDst>(output_data, dst_opencv.ptr<float>(), output_height, output_width, outStride, diff_THR);
#endif

    free(input_data);
    free(input_data_opencv);
    free(output_data);
    free(output_data_opencv);
    return 0;
}


TEST_F(halfBilateralFilterTest, float){
    imageParam p ;
    ASSERT_TRUE(!imageparams.empty());
    for(int i = 0;i<imageparams.size();i++)
    {
        P = imageparams[i];
        testBilateralFilterFunc<half_t, 1, 1, 1, BORDER_DEFAULT, 255>(p, 10e-5);
        testBilateralFilterFunc<half_t, 3, 3, 3, BORDER_DEFAULT, 255>(p, 10e-5);
    }
}

#undef PRINT_ALL
#undef N




