#include "test_head.hpp"
using namespace HPC::fastcv;
//#define PRINT_ALL
struct imageParam{
    size_t srcWidth;
    size_t srcHeight;
    size_t dstWidth;
    size_t dstHeight;
    int spadding;
    int dpadding;
};




class integralTest : public testing::Test
{
     public:
        virtual void SetUp()
        {
            std::vector<imageParam>::iterator iter = imageparams.begin();
            if(globalEnvironment::Instance()->testconfig.hasSection("integralTest"))
            {
                Cfg::cfg_type vec_arg = globalEnvironment::Instance()->testconfig.sectionConfigVec("integralTest");
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
                imagep={100,100, 100, 100, 7, 11};
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
        tmp = rand()% 255;
        data[i] = (T) ( (float)tmp/(float)val );
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

template<typename T, int nc>
void checkResult(const T *data1, const T *data2, const T * input_data, const int height, const int width, const float diff_THR) {
    double max = DBL_MIN;
    double min = DBL_MAX;
    double temp_diff = 0.0;

    int num = height*width *nc;
    for(int i=0; i < num; i++){
        //ASSERT_NEAR(data1[i], data2[i], 1e-5);
        temp_diff = fabs((double) *(data1 + i)-(double) *(data2 + i));
#ifdef PRINT_ALL
		//if(temp_diff > diff_THR)
        std::cout << "Print data12: " << (float) data1[i] << " : " << (float) data2[i] << " : " << temp_diff << std::endl;
#endif
        max = (temp_diff > max) ? temp_diff : max;
        min = (temp_diff < min) ? temp_diff : min;
    }
    EXPECT_LT(max, diff_THR);
    //ASSERT_NEAR(order, 1e-5, 1e-5);
}

template<typename T, int nc>
void checkResult(const T *data1, const T *data2, const T * input_data, const int height, const int width, int dstep1, int dstep2, const float diff_THR) {
    double max = DBL_MIN;
    double min = DBL_MAX;
    double temp_diff = 0.0;

    int num = height*width *nc;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width*nc; j++){
            float val1 = data1[i * dstep1 +j];
            float val2 = data2[i * dstep2 + j];

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

/*
int main(int argc, char *argv[]){
    if(argc != 1 && argc != 7){
        printf("Error params!\n");
        printf("Usage: ./run src_width src_height \
                dst_width dst_height\n");
    }
   // imagep={64, 64, 64, 64};
  //  imagep={10, 10, 10, 10};
    imagep={100,100, 100, 100, 7, 11};
    if(argc == 7){
        imagep.srcWidth = atoi(argv[1]);
        imagep.srcHeight = atoi(argv[2]);
        imagep.dstWidth = atoi(argv[3]);
        imagep.dstHeight = atoi(argv[4]);
        imagep.spadding = atoi(argv[5]);
        imagep.dpadding = atoi(argv[6]);
        if(imagep.srcWidth != imagep.dstWidth || imagep.srcHeight != imagep.dstHeight)
        {
            printf("Error params: Dst Width and Height should be equal to Src Width and Height plus one, respectively!");
            exit(-1);
        }
    }

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
*/
template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc, int val_div>
int testIntegralFunc(imageParam imagep, float diff_THR) {
    size_t input_width = imagep.srcWidth;
    size_t input_height = imagep.srcHeight;
    size_t input_channels = ncSrc;
    size_t output_width = imagep.dstWidth;
    size_t output_height = imagep.dstHeight;
    size_t output_channels = ncDst;
    int spadding = imagep.spadding;
    int dpadding = imagep.dpadding;


    printf("Input image scale: width*height*channels = %ld*%ld*%ld\n", input_width, input_height, input_channels);
    printf("Output image scale: width*height*channels = %ld*%ld*%ld\n", output_width, output_height, output_channels);

/*    size_t input_data_num = input_width*input_height*input_channels;
    Tsrc* input_data = (Tsrc*)malloc(input_data_num*sizeof(Tsrc));
    randomRangeData<Tsrc, val_div>(input_data, input_data_num);*/

    size_t step = (input_width * input_channels + spadding)*sizeof(Tsrc);
    Tsrc* input_data = (Tsrc*)malloc(input_height*step);
    randomPaddingData<Tsrc, val_div>(input_data, input_height, input_width*input_channels, input_width * input_channels + spadding);


   /* std::cout << " Original data:" << std::endl;
    for(size_t i = 0; i < input_data_num; i++)
        std::cout << i << " : " << (float) input_data[i] << " " << std::endl;*/


#if defined(FASTCV_USE_CUDA)
    Mat<Tsrc, ncSrc> src(input_height, input_width, step);
    src.fromHost(input_data);

    const Mat<Tsrc, ncSrc,EcoEnv_t::ECO_ENV_X86> src_x86(input_height, input_width, input_data, step);

#elif defined(FASTCV_USE_OCL)
    Mat<Tsrc, ncSrc> src(input_height, input_width, step);
    src.fromHost(input_data);
#else
    const Mat<Tsrc, ncSrc> src(input_height, input_width, input_data, step);
#endif

    size_t dstep = (output_width * output_channels + dpadding)*sizeof(Tdst);
    Tdst* output_data = (Tdst*)malloc(output_height*dstep);
    Tdst* output_data_opencv = (Tdst*)malloc(output_height*dstep);
    randomPaddingData<Tdst, val_div>(output_data, output_height,
            output_width*output_channels, output_width * output_channels + dpadding);

    randomPaddingData<Tdst, val_div>(output_data_opencv, output_height,
        output_width*output_channels, output_width * output_channels + dpadding);


#if defined(FASTCV_USE_CUDA)
    Mat<Tdst, ncDst> dst(output_height, output_width, dstep);
    dst.fromHost(output_data);

    Mat<Tdst, ncDst,EcoEnv_t::ECO_ENV_X86> dst_x86(output_height, output_width, output_data, dstep);
#elif defined(FASTCV_USE_OCL)
    Mat<Tdst, ncDst> dst(output_height, output_width, dstep);
    dst.fromHost(output_data);
#else
    Mat<Tdst, ncDst> dst(output_height, output_width, output_data, dstep);
#endif



#if defined(FASTCV_USE_CUDA)
    // call opencv to verify result
    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, ncSrc), input_data, step);
    cv::Mat dst_opencv(output_height+1, output_width+1, CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst));
    cv::integral(src_opencv, dst_opencv, CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst));

    cv::Mat dst_opencv1(output_height, output_width, CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst));
    for(int i = 0; i < output_height; i++)
    {
        memcpy(dst_opencv1.ptr<Tdst>()+i*output_width*nc, dst_opencv.ptr<Tdst>()+(i+1)*(output_width+1)*nc + nc, output_width*nc*sizeof(Tdst));
    }

    // call fastcv
    HPCStatus_t sta = integral<Tsrc, ncSrc, Tdst, ncDst, nc>(src, &dst);
    //X86
    HPCStatus_t sta_x86 = integral<Tsrc, ncSrc, Tdst, ncDst, nc>(src_x86, &dst_x86);
    EXPECT_EQ(HPC_SUCCESS, sta);
    EXPECT_EQ(HPC_SUCCESS, sta_x86);
#elif defined(FASTCV_USE_OCL)
    // call opencv to verify result
    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, ncSrc), input_data, step);
    cv::Mat dst_opencv(output_height+1, output_width+1, CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst));
    cv::integral(src_opencv, dst_opencv, CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst));

    cv::Mat dst_opencv1(output_height, output_width, CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst));
    for(int i = 0; i < output_height; i++)
    {
        memcpy(dst_opencv1.ptr<Tdst>()+i*output_width*nc, dst_opencv.ptr<Tdst>()+(i+1)*(output_width+1)*nc + nc, output_width*nc*sizeof(Tdst));
    }

    // call fastcv
    HPCStatus_t sta = integral<Tsrc, ncSrc, Tdst, ncDst, nc>(src, &dst);
    EXPECT_EQ(HPC_SUCCESS, sta);
#else

    // call opencv to verify result
    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, ncSrc), input_data, step);
    cv::Mat dst_opencv(output_height+1, output_width+1, CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst));
    cv::integral(src_opencv, dst_opencv, CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst));

    cv::Mat dst_opencv1(output_height, output_width, CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst));
    for(int i = 0; i < output_height; i++)
    {
        memcpy(dst_opencv1.ptr<Tdst>()+i*output_width*nc, dst_opencv.ptr<Tdst>()+(i+1)*(output_width+1)*nc + nc, output_width*nc*sizeof(Tdst));
    }

    // call fastcv
    HPCStatus_t sta = integral<Tsrc, ncSrc, Tdst, ncDst, nc>(src, &dst);
    EXPECT_EQ(HPC_SUCCESS, sta);
#endif


    // check results
#if defined(FASTCV_USE_CUDA)
    //size_t output_data_num = output_width*output_height*output_channels;
    //Tdst* dst_data_x86 = (Tdst*)malloc(output_data_num*sizeof(Tdst));
    dst.toHost(output_data);
    checkResult<Tdst, ncDst>(output_data, dst_opencv1.ptr<Tdst>(),input_data, output_height, output_width,
        dstep/sizeof(Tdst), dst_opencv1.step/sizeof(Tdst), diff_THR);

     checkResult<Tdst, ncDst>((Tdst*)dst_x86.ptr(), dst_opencv1.ptr<Tdst>(), input_data, output_height, output_width, dstep/sizeof(Tdst), dst_opencv1.step/sizeof(Tdst), diff_THR);
    //free(dst_data_x86);
#elif defined(FASTCV_USE_OCL)
    //size_t output_data_num = output_width*output_height*output_channels;
    //Tdst* dst_data_x86 = (Tdst*)malloc(output_data_num*sizeof(Tdst));
    dst.toHost(output_data);
    checkResult<Tdst, ncDst>(output_data, dst_opencv1.ptr<Tdst>(), input_data, output_height, output_width, dstep/sizeof(Tdst), dst_opencv1.step/sizeof(Tdst), diff_THR);
    //free(dst_data_x86);
#else
    checkResult<Tdst, ncDst>((Tdst*)dst.ptr(), dst_opencv1.ptr<Tdst>(), input_data, output_height, output_width, dstep/sizeof(Tdst), dst_opencv1.step/sizeof(Tdst), diff_THR);
#endif

    return 0;
}

/*TEST_P(IntegralParamTest, uchar_uint){
    imageParam p = GetParam();
    testIntegralFunc<uchar, 1, uint, 1, 1, 1>(p, 1e-5);
    testIntegralFunc<uchar, 3, uint, 3, 3, 1>(p, 1e-5);
    testIntegralFunc<uchar, 4, uint, 4, 4, 1>(p, 1e-5);
} */

TEST_F(integralTest, float_float){
    imageParam p ;
  //  ASSERT_TRUE(!imageparams.empty());
  //  for(int i =0; i<imageparams.size(); i++)
  //  {
        p = imagep;
         testIntegralFunc<float, 1, float, 1, 1, 255>(p, 1e-5);
        testIntegralFunc<float, 3, float, 3, 3, 255>(p, 1e-5);
        testIntegralFunc<float, 4, float, 4, 4, 255>(p, 1e-5);
 //   }
   
}

/*TEST_P(IntegralParamTest, double_double){
    imageParam p = GetParam();
    testIntegralFunc<double, 1, double, 1, 1, 255>(p, 0.1);
    testIntegralFunc<double, 3, double, 3, 3, 255>(p, 0.1);
    testIntegralFunc<double, 4, double, 4, 4, 255>(p, 0.1);
} */

