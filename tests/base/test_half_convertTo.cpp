#include "test_head.hpp"
using namespace HPC::fastcv;
using namespace uni::half;
#define PRINT_ALL
struct imageParam{
    size_t srcWidth;
    size_t srcHeight;
    size_t dstWidth;
    size_t dstHeight;
    int spadding;
    int dpadding;
};



class halfConverToTest : public testing::Test
{
    public:
        virtual void SetUp()
        {
            std::vector<imageParam>::iterator iter = imageparams.begin();
            if(globalEnvironment::Instance()->testconfig.hasSection("halfConverToTest"))
            {
                Cfg::cfg_type vec_arg = globalEnvironment::Instance()->testconfig.sectionConfigVec("halfConverToTest");
                int num = vec_arg.size();
                for(int i = 0;i < num; i++)
                {
                    ASSERT_TRUE(vec_arg[i].find("srcHeight")!=vec_arg[i].end())<<"argument:srcHeight not found";
                    ASSERT_TRUE(vec_arg[i].find("srcWidth")!=vec_arg[i].end())<<"argument:srcWidth not found";
                    ASSERT_TRUE(vec_arg[i].find("dstWidth")!=vec_arg[i].end())<<"argument:dstWidth not found";
                    ASSERT_TRUE(vec_arg[i].find("dstHeight")!=vec_arg[i].end())<<"argument:dstHeight not found";                   
                    ASSERT_TRUE(vec_arg[i].find("spadding")!=vec_arg[i].end())<<"argument:spadding not found";
                    ASSERT_TRUE(vec_arg[i].find("dpadding")!=vec_arg[i].end())<<"argument:dpadding not found";
                    imagep.srcHeight = std::stoul(vec_arg[i]["srcHeight"]);
                    imagep.srcWidth = std::stoul(vec_arg[i]["srcWidth"]);
                    imagep.dstWidth = std::stoul(vec_arg[i]["dstWidth"]);
                    imagep.dstHeight = std::stoul(vec_arg[i]["dstHeight"]);              
                    imagep.spadding = std::stoi(vec_arg[i]["spadding"]);
                    imagep.dpadding = std::stoi(vec_arg[i]["dpadding"]);
                    iter = imageparams.insert(iter,imagep);

                }
            }
            else
            {
                imagep={64, 64, 64, 64,7,11};
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

template<typename T, int val>
void randomRangeData(T *data, const size_t num){
    size_t tmp;
    clock_t ct = clock();
    srand((unsigned int)ct);

    for(size_t i = 0; i < num; i++){
        tmp = rand() % 255; //max;
        data[i] = (T) ( (float) tmp / (float) val );
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
               // std::cout<<data[i*widthStride + j]<<std::endl;
          }
        }
}

template<typename T, int nc>
void checkResult(const T *data1, const float *data2, const int height, const int width, int dstep, int dstep1, const float diff_THR) {
    double max = DBL_MIN;
    double min = DBL_MAX;
    double temp_diff = 0.0;

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width*nc; j++){
            float val1 = data1[i * dstep +j];
            float val2 = data2[i * dstep1 + j];

            temp_diff = fabs(val1 - val2);

#ifdef PRINT_ALL
            if(temp_diff > diff_THR)
            std::cout << "Print data12: " << i << " : " <<  j << " : "<< (float) val1 << " : " << (float) val2 << " : " << temp_diff << std::endl;
#endif
            max = (temp_diff > max) ? temp_diff : max;
            min = (temp_diff < min) ? temp_diff : min;
        }
    }
    EXPECT_LT(max, diff_THR);
}
template<typename T, int nc>
void checkResult(const T *data1, const T *data2, const int height, const int width, int dstep, int dstep1, const float diff_THR) {
    double max = DBL_MIN;
    double min = DBL_MAX;
    double temp_diff = 0.0;

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width*nc; j++){
            float val1 = data1[i * dstep +j];
            float val2 = data2[i * dstep1 + j];

            temp_diff = fabs(val1 - val2);

#ifdef PRINT_ALL
            if(temp_diff > diff_THR)
            std::cout << "Print data12: " << i << " : " <<  j << " : "<< (float) val1 << " : " << (float) val2 << " : " << temp_diff << std::endl;
#endif
            max = (temp_diff > max) ? temp_diff : max;
            min = (temp_diff < min) ? temp_diff : min;
        }
    }
    EXPECT_LT(max, diff_THR);
}

template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc, int val_div>
int testConverToFunc(imageParam imagep,float scale, float diff_THR) {
    size_t input_width = imagep.srcWidth;
    size_t input_height = imagep.srcHeight;
    size_t input_channels = nc;
    size_t output_width = imagep.dstWidth;
    size_t output_height = imagep.dstHeight;
    size_t output_channels = nc;
    int spadding = imagep.spadding;
    int dpadding = imagep.dpadding;

    printf("Input Mat scale: width*height*channels = %ld * %ld * %ld\n", input_width, input_height, input_channels);
    printf("Output Mat scale: width*height*channels = %ld * %ld * %ld\n", output_width, output_height, output_channels);

    int inStride = input_width * input_channels + spadding;
    size_t step = (inStride)*sizeof(Tsrc);
    Tsrc* input_data = (Tsrc*)malloc(input_height*step);
    randomPaddingData<Tsrc, val_div>(input_data, input_height,
            input_width*input_channels, input_width * input_channels + spadding);

#if defined(USE_CUDA)
    Mat<Tsrc, ncSrc> src(input_height, input_width, step);
    src.fromHost(input_data);
#elif defined(USE_OCL)
    Mat<Tsrc, ncSrc> src(input_height, input_width, step);
    src.fromHost(input_data);
#else
    const Mat<Tsrc, ncSrc> src(input_height, input_width, input_data, step);
#endif


    int outStride = output_width * output_channels + dpadding;
    size_t dstep = (outStride)*sizeof(Tdst);
    size_t dstep_opencv = (outStride)*sizeof(float);
    Tdst* output_data = (Tdst*)malloc(output_height*dstep);
    float* output_data_opencv = (float*)malloc(output_height*dstep_opencv);

    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, ncSrc), input_data, step);

#if defined(USE_CUDA)
    Mat<Tdst, ncDst> dst(output_height, output_width, dstep);
    dst.fromHost(output_data);
#elif defined(USE_OCL)
    Mat<Tdst, ncDst> dst(output_height, output_width, dstep);
    dst.fromHost(output_data);
#else
    Mat<Tdst, ncDst> dst(output_height, output_width, output_data, dstep);
#endif

#if defined(USE_CUDA)
    HPCStatus_t sta = convertTo<Tsrc, ncSrc, Tdst, ncDst, nc>(src, scale, &dst);
    EXPECT_EQ(HPC_SUCCESS, sta);
    int opencv_type = 0;
    if(nc == 1){
        opencv_type = (sizeof(Tdst) == sizeof(float))? CV_32FC1 :CV_8UC1;
    }
    if(nc == 3){
        opencv_type = (sizeof(Tdst) == sizeof(float))? CV_32FC3 :CV_8UC3;
    }
    cv::Mat dst_opencv(output_height, output_width, opencv_type, output_data_opencv, dstep_opencv);
    std::cout<<"dstep: "<<dst_opencv.step<<std::endl;
    src_opencv.convertTo(dst_opencv, CV_MAKETYPE(cv::DataType<float>::depth, ncDst), scale);
    std::cout<<"dstep: "<<dst_opencv.step<<std::endl;
#elif defined(USE_OCL)
    HPCStatus_t sta = convertTo<Tsrc, ncSrc, Tdst, ncDst, nc>(src, scale, &dst);
    EXPECT_EQ(HPC_SUCCESS, sta);
    int opencv_type = 0;
    if(nc == 1){
        opencv_type = (sizeof(Tdst) == sizeof(float))? CV_32FC1 :CV_8UC1;
    }
    if(nc == 3){
        opencv_type = (sizeof(Tdst) == sizeof(float))? CV_32FC3 :CV_8UC3;
    }
    cv::Mat dst_opencv(output_height, output_width, opencv_type, output_data_opencv, dstep);
    std::cout<<"dstep: "<<dst_opencv.step<<std::endl;;
    src_opencv.convertTo(dst_opencv, CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst), scale);
    std::cout<<"dstep: "<<dst_opencv.step<<std::endl;;

#else
    HPCStatus_t sta = convertTo<Tsrc, ncSrc, Tdst, ncDst, nc>(src, scale, &dst);
    EXPECT_EQ(HPC_SUCCESS, sta);
    int opencv_type = 0;
    if(nc == 1){
        opencv_type = (sizeof(Tdst) == sizeof(float))? CV_32FC1 :CV_8UC1;
    }
    if(nc == 3){
        opencv_type = (sizeof(Tdst) == sizeof(float))? CV_32FC3 :CV_8UC3;
    }
    cv::Mat dst_opencv(output_height, output_width, opencv_type, output_data_opencv, dstep);
    std::cout<<"dstep: "<<dst_opencv.step<<std::endl;;
    src_opencv.convertTo(dst_opencv, CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst), scale);
    std::cout<<"dstep: "<<dst_opencv.step<<std::endl;;

#endif



#if defined(USE_CUDA)
    dst.toHost(output_data);
    int dst_opencv_step = dst_opencv.step;
    checkResult<Tdst, ncDst>(output_data, dst_opencv.ptr<float>(), output_height, output_width, dstep/sizeof(Tdst),dst_opencv_step/sizeof(float) , diff_THR);
#elif defined(USE_OCL)
    dst.toHost(output_data);
    checkResult<Tdst, ncDst>(output_data, dst_opencv.ptr<float>(), output_height, output_width, dstep/sizeof(Tdst), dst_opencv.step/sizeof(Tdst), diff_THR);
#endif

    free(input_data);
    free(output_data);
    free(output_data_opencv);
    return 0;
}





template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc, int val_div>
int testConverToFunc2(imageParam imagep,float scale, float diff_THR) {
    size_t input_width = imagep.srcWidth;
    size_t input_height = imagep.srcHeight;
    size_t input_channels = nc;
    size_t output_width = imagep.dstWidth;
    size_t output_height = imagep.dstHeight;
    size_t output_channels = nc;
    int spadding = imagep.spadding;
    int dpadding = imagep.dpadding;

    printf("Input Mat scale: width*height*channels = %ld * %ld * %ld\n", input_width, input_height, input_channels);
    printf("Output Mat scale: width*height*channels = %ld * %ld * %ld\n", output_width, output_height, output_channels);

    int inStride = input_width * input_channels + spadding;
    size_t step = (inStride)*sizeof(Tsrc);
    Tsrc* input_data = (Tsrc*)malloc(input_height*step);
    randomPaddingData<Tsrc, val_div>(input_data, input_height,
            input_width*input_channels, inStride);

    size_t step_opencv = inStride * sizeof(float);
    float* input_data_opencv = (float*)malloc(input_height*step_opencv);
    for(int i=0;i<input_height * inStride;i++){
        input_data_opencv[i] = input_data[i];
    }

#if defined(USE_CUDA)
    Mat<Tsrc, ncSrc> src(input_height, input_width, step);
    src.fromHost(input_data);
#elif defined(USE_OCL)
    Mat<Tsrc, ncSrc> src(input_height, input_width, step);
    src.fromHost(input_data);
#else
    const Mat<Tsrc, ncSrc> src(input_height, input_width, input_data, step);
#endif


    int outStride = output_width * output_channels + dpadding;
    size_t dstep = (outStride)*sizeof(Tdst);
    size_t dstep_opencv = (outStride)*sizeof(Tdst);
    Tdst* output_data = (Tdst*)malloc(output_height*dstep);
    Tdst* output_data_opencv = (Tdst*)malloc(output_height*dstep_opencv);

    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<float>::depth, ncSrc), input_data_opencv, step_opencv);

#if defined(USE_CUDA)
    Mat<Tdst, ncDst> dst(output_height, output_width, dstep);
    dst.fromHost(output_data);
#elif defined(USE_OCL)
    Mat<Tdst, ncDst> dst(output_height, output_width, dstep);
    dst.fromHost(output_data);
#else
    Mat<Tdst, ncDst> dst(output_height, output_width, output_data, dstep);
#endif

#if defined(USE_CUDA)
    HPCStatus_t sta = convertTo<Tsrc, ncSrc, Tdst, ncDst, nc>(src, scale, &dst);
    EXPECT_EQ(HPC_SUCCESS, sta);
    int opencv_type = 0;
    if(nc == 1){
        opencv_type = (sizeof(Tdst) == sizeof(float))? CV_32FC1 :CV_8UC1;
    }
    if(nc == 3){
        opencv_type = (sizeof(Tdst) == sizeof(float))? CV_32FC3 :CV_8UC3;
    }
    cv::Mat dst_opencv(output_height, output_width, opencv_type, output_data_opencv, dstep_opencv);
    std::cout<<"dstep: "<<dst_opencv.step<<std::endl;
    src_opencv.convertTo(dst_opencv, CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst), scale);
    std::cout<<"dstep: "<<dst_opencv.step<<std::endl;

#elif defined(USE_OCL)
    HPCStatus_t sta = convertTo<Tsrc, ncSrc, Tdst, ncDst, nc>(src, scale, &dst);
    EXPECT_EQ(HPC_SUCCESS, sta);
    int opencv_type = 0;
    if(nc == 1){
        opencv_type = (sizeof(Tdst) == sizeof(float))? CV_32FC1 :CV_8UC1;
    }
    if(nc == 3){
        opencv_type = (sizeof(Tdst) == sizeof(float))? CV_32FC3 :CV_8UC3;
    }
    cv::Mat dst_opencv(output_height, output_width, opencv_type, output_data_opencv, dstep);
    std::cout<<"dstep: "<<dst_opencv.step<<std::endl;;
    src_opencv.convertTo(dst_opencv, CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst), scale);
    std::cout<<"dstep: "<<dst_opencv.step<<std::endl;;

#else

#endif



#if defined(USE_CUDA)
    dst.toHost(output_data);
    int dst_opencv_step = dst_opencv.step;
    checkResult<Tdst, ncDst>(output_data, dst_opencv.ptr<Tdst>(), output_height, output_width, dstep/sizeof(Tdst),dst_opencv_step/sizeof(Tdst) , diff_THR);
#elif defined(USE_OCL)
    dst.toHost(output_data);
    checkResult<Tdst, ncDst>(output_data, dst_opencv.ptr<Tdst>(), output_height, output_width, dstep/sizeof(Tdst), dst_opencv.step/sizeof(Tdst), diff_THR);
#endif

    free(input_data);
    free(input_data_opencv);
    free(output_data);
    free(output_data_opencv);
    return 0;
}

TEST_F(halfConverToTest, uchar_float){
    imageParam p;
    ASSERT_TRUE(!imageparams.empty());
    for(int i = 0; i<imageparams.size(); i++)
    {
        p = imageparams[i];
        testConverToFunc<uchar, 1, half_t, 1, 1, 1>(p,1.0/255.0, 1e-3);
        testConverToFunc<uchar, 3, half_t, 3, 3, 1>(p,1.0/255.0, 1e-3);
        testConverToFunc<uchar, 4, half_t, 4, 4, 1>(p,1.0/255.0, 1e-3);
    }
   
}

TEST_F(halfConverToTest, float_uchar){
    imageParam p;
    ASSERT_TRUE(!imageparams.empty());
    for(int i = 0; i<imageparams.size(); i++)
    {
        p = imageparams[i];
        testConverToFunc2<half_t, 1, uchar, 1, 1, 255>(p,255.0, 0.1);
        testConverToFunc2<half_t, 3, uchar, 3, 3, 255>(p,255.0, 0.1);
        testConverToFunc2<half_t, 4, uchar, 4, 4, 255>(p,255.0, 0.1);
    }
}
