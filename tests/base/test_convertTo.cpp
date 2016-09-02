#include "test_head.hpp"
#include "opencv2/opencv.hpp"
using namespace HPC::fastcv;

#define PRINT_ALL
struct imageParam{
    size_t srcWidth;
    size_t srcHeight;
    size_t dstWidth;
    size_t dstHeight;
    int spadding;
    int dpadding;
};


class ConverToTest : public testing::Test
{
     public:
        virtual void SetUp()
        {
             std::vector<imageParam>::iterator iter = imageparams.begin();
            if(globalEnvironment::Instance()->testconfig.hasSection("ConverToTest"))
            {
                Cfg::cfg_type vec_arg = globalEnvironment::Instance()->testconfig.sectionConfigVec("ConverToTest");
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
                imagep={64, 64, 64, 64, 7, 11};
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
                //printf("%d\n",data[i]);
            }
        }
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
/*
int main(int argc, char *argv[]){
    if(argc != 1 && argc != 7){
        printf("Error params!\n");
        printf("Usage: ./run src_width src_height \
                dst_width dst_height \n");
    }
    imagep={64, 64, 64, 64, 7, 11};
    //imagep={2355, 2355, 2355, 2355,9 ,19};
    imagep={255, 355, 255, 355,8 ,16};
    //imagep={161, 20, 161, 20,0,0};
    if(argc == 7){
        imagep.srcWidth = atoi(argv[1]);
        imagep.srcHeight = atoi(argv[2]);
        imagep.dstWidth = atoi(argv[3]);
        imagep.dstHeight = atoi(argv[4]);
        imagep.spadding = atoi(argv[5]);
        imagep.dpadding = atoi(argv[6]);
    }

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
*/
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

/*    size_t input_data_num = input_width*input_height*input_channels;
    Tsrc* input_data = (Tsrc*)malloc(input_data_num*sizeof(Tsrc));

    randomRangeData<Tsrc, val_div>(input_data, input_data_num);*/
    size_t step = (input_width * input_channels + spadding)*sizeof(Tsrc);
    Tsrc* input_data = (Tsrc*)malloc(input_height*step);
    randomPaddingData<Tsrc, val_div>(input_data, input_height,
            input_width*input_channels, input_width * input_channels + spadding);

/*
for(int i=0;i<input_data_num;i++){
	printf(" %d  %f\n",i,(float)input_data[i]);
}*/

#if defined(FASTCV_USE_CUDA)
    Mat<Tsrc, ncSrc> src(input_height, input_width, step);
    src.fromHost(input_data);

    const Mat<Tsrc, ncSrc, EcoEnv_t::ECO_ENV_X86> src_x86(input_height, input_width, input_data, step);

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

    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, ncSrc), input_data, step);

#if defined(FASTCV_USE_CUDA)
    Mat<Tdst, ncDst> dst(output_height, output_width, dstep);
    dst.fromHost(output_data);

    Mat<Tdst, ncDst, EcoEnv_t::ECO_ENV_X86> dst_x86(output_height, output_width, output_data, dstep);

#elif defined(FASTCV_USE_OCL)
    Mat<Tdst, ncDst> dst(output_height, output_width, dstep);
    dst.fromHost(output_data);
#else
    Mat<Tdst, ncDst> dst(output_height, output_width, output_data, dstep);
#endif

#if defined(FASTCV_USE_CUDA)
    HPCStatus_t sta = convertTo<Tsrc, ncSrc, Tdst, ncDst, nc>(src, scale, &dst);
    EXPECT_EQ(HPC_SUCCESS, sta);

    HPCStatus_t sta_x86 = convertTo<Tsrc, ncSrc, Tdst, ncDst, nc>(src_x86, scale, &dst_x86);
    EXPECT_EQ(HPC_SUCCESS,sta_x86);

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
#elif defined(FASTCV_USE_OCL)
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



#if defined(FASTCV_USE_CUDA)
    //size_t output_data_num = output_width*output_height*output_channels;
    //Tdst* dst_data_x86 = (Tdst*)malloc(output_data_num*sizeof(Tdst));
    dst.toHost(output_data);
    checkResult<Tdst, ncDst>(output_data, dst_opencv.ptr<Tdst>(), output_height, output_width, dstep/sizeof(Tdst), dst_opencv.step/sizeof(Tdst), diff_THR);
        
    checkResult<Tdst, ncDst>((Tdst*)dst_x86.ptr(), dst_opencv.ptr<Tdst>(), output_height, output_width, dstep/sizeof(Tdst), dst_opencv.step/sizeof(Tdst), diff_THR);

#elif defined(FASTCV_USE_OCL)
    //size_t output_data_num = output_width*output_height*output_channels;
    //Tdst* dst_data_x86 = (Tdst*)malloc(output_data_num*sizeof(Tdst));
    dst.toHost(output_data);
    checkResult<Tdst, ncDst>(output_data, dst_opencv.ptr<Tdst>(), output_height, output_width, dstep/sizeof(Tdst), dst_opencv.step/sizeof(Tdst), diff_THR);
#else
    checkResult<Tdst, ncDst>((Tdst*)dst.ptr(), dst_opencv.ptr<Tdst>(), output_height, output_width, dstep/sizeof(Tdst), dst_opencv.step/sizeof(Tdst), diff_THR);
#endif

    free(input_data);
    free(output_data);
    free(output_data_opencv);
    return 0;
}

TEST_F(ConverToTest, uchar_float){
    imageParam p ;
 //   ASSERT_TRUE(!imageparams.empty());
 //   for(int i = 0; i < imageparams.size(); i++)
 //   {
        p = imagep;
        testConverToFunc<uchar, 1, float, 1, 1, 1>(p,1.0/255.0, 1e-5);
        testConverToFunc<uchar, 3, float, 3, 3, 1>(p,1.0/255.0, 1e-5);
        testConverToFunc<uchar, 4, float, 4, 4, 1>(p,1.0/255.0, 1e-5);
        testConverToFunc<float, 1, uchar, 1, 1, 255>(p,255.0, 0.1);
        testConverToFunc<float, 3, uchar, 3, 3, 255>(p,255.0, 0.1);
        testConverToFunc<float, 4, uchar, 4, 4, 255>(p,255.0, 0.1);
  //  }
   
}

