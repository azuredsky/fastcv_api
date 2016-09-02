#include <test_head.hpp>
#include <opencv2/gpu/gpu.hpp>
#include<opencv2/ocl/ocl.hpp>
using namespace HPC::fastcv;
#define N 1000

#define PRINT_ALL
struct imageParam{
    size_t srcWidth;
    size_t srcHeight;
    size_t dstWidth;
    size_t dstHeight;
    int spadding;
    int dpadding;
};


//ColorCvtType type1 = BGR2I420;
//ColorCvtType type2 = I4202BGR;
//ColorCvtType type3 = YUV2GRAY_420;

class I420cvtColorTest : public testing::Test
{
     public:
        virtual void SetUp()
        {
            std::vector<imageParam>::iterator iter = imageparams.begin();

            if(globalEnvironment::Instance()->testconfig.hasSection("I420cvtColorTest"))
            {
                Cfg::cfg_type vec_arg = globalEnvironment::Instance()->testconfig.sectionConfigVec("I420cvtColorTest");
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
               imagep={1920, 1080, 1920, 1080, 7, 11};
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
void checkResult_colortype(const T *data1, const T *data2, const int height,
        const int width, int dstep, int dstep1, const float diff_THR, ColorCvtType type) {
    double max = DBL_MIN;
    double min = DBL_MAX;
    double ave_diff = 0.0;
    double l1_diff  = 0.0;
    double l2_diff  = 0.0;
    double temp_diff = 0.0;
    double ave_data = 0.0;
    double order = 0.0;

    int num = height*width *nc;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width * nc; j++){
            float val1 = data1[i * dstep + j];
            float val2 = data2[i * dstep1 + j];
            temp_diff = fabs(val1 - val2);

#ifdef PRINT_ALL

            if(temp_diff>diff_THR)
                std::cout << "Print data12: " << i << " : " << j << " : " <<
                    (float) data1[i * dstep + j] << " : " << (float) data2[i * dstep1 + j] << " : " << temp_diff << std::endl;
#endif
            max = (temp_diff > max) ? temp_diff : max;
            min = (temp_diff < min) ? temp_diff : min;
            ave_data += fabs((double)data1[i]);
            l1_diff += temp_diff;
            l2_diff += temp_diff*temp_diff;
        }
    }
    ave_data = ave_data / num;
    ave_diff = l1_diff / num;
    order = (ave_data == 0) ? 0.0 : ave_diff / ave_data;
    l2_diff = sqrt(l2_diff);
    EXPECT_LT(max, diff_THR);
}

/*
int main(int argc, char *argv[]){
    if(argc != 1 && argc != 7){
        printf("Error params!\n");
        printf("Usage: ./run src_width src_height dst_width dst_height \n");
    }
     //imagep={10, 10, 10, 10, 7, 11};
     imagep={1920, 1080, 1920, 1080, 7, 11};
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
template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, ColorCvtType type, int val_div>
int testCvtColorFunc(imageParam imagep,float diff_THR) {
    size_t input_width = imagep.srcWidth;
    size_t input_height = imagep.srcHeight;
    size_t input_channels = ncSrc;
    size_t output_width = imagep.dstWidth;
    size_t output_height = imagep.dstHeight;
    size_t output_channels = ncDst;
    int spadding = imagep.spadding;
    int dpadding = imagep.dpadding;

    printf("Input image scale: width*height*channels = %ld * %ld * %ld\n", input_width, input_height, input_channels);
    printf("Output image scale: width*height*channels = %ld * %ld * %ld\n", output_width, output_height, output_channels);


    if(type == I4202BGR || type == YUV2GRAY_420)
        input_height = input_height * 3 / 2;
    if(type == BGR2I420)
        output_height = output_height * 3 / 2;
    size_t step = (input_width * input_channels + spadding)*sizeof(Tsrc);
    Tsrc* input_data = (Tsrc*)malloc(input_height*step);
    Tsrc* input_data1 = (Tsrc*)malloc(input_height*step);
    randomPaddingData<Tsrc, val_div>(input_data, input_height,
            input_width*input_channels, input_width * input_channels + spadding);
    randomPaddingData<Tsrc, val_div>(input_data1, input_height,
            input_width*input_channels, input_width * input_channels + spadding);


    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, ncSrc), input_data, step);

    size_t dstep = (output_width * output_channels + dpadding)*sizeof(Tdst);
    Tdst* output_data = (Tdst*)malloc(output_height * dstep);
    Tdst* output_data_opencv = (Tdst*)malloc(output_height*dstep);
    //randomPaddingData<Tdst, val_div>(output_data, output_height,
      //      output_width*output_channels, output_width * output_channels + dpadding);

   // randomPaddingData<Tdst, val_div>(output_data_opencv, output_height,
     //   output_width*output_channels, output_width * output_channels + dpadding);

#if defined(FASTCV_USE_CUDA)
    //const Mat<Tsrc, ncSrc> src(input_height, input_width, step);
	//src.fromHost(input_data);
    Mat<Tsrc, ncSrc> src(input_height, input_width, step);
    src.fromHost(input_data);

    Mat<Tdst, ncDst> dst(output_height, output_width, dstep);
   // dst.fromHost(input_data);
	HPCStatus_t sta = cvtColor<type, Tsrc, ncSrc, Tdst, ncDst>(src, &dst);
    EXPECT_EQ(HPC_SUCCESS, sta);
   //X86
    const Mat<Tsrc, ncSrc,EcoEnv_t::ECO_ENV_X86> src_x86(input_height, input_width, input_data,step);


    Mat<Tdst, ncDst,EcoEnv_t::ECO_ENV_X86> dst_x86(output_height, output_width,output_data, dstep);


    HPCStatus_t sta_x86 = cvtColor<type, Tsrc, ncSrc, Tdst, ncDst>(src_x86, &dst_x86);
    EXPECT_EQ(HPC_SUCCESS, sta_x86);

    cv::Mat dst_opencv(output_height, output_width, CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst), output_data_opencv, dstep);
    double time_fastcv;
    switch(type) {
         case BGR2I420:
            if(ncSrc == 4) cv::cvtColor(src_opencv, dst_opencv, CV_BGR2BGRA);
            cv::cvtColor(src_opencv, dst_opencv, CV_BGR2YUV_IYUV);
            break;

        case I4202BGR:
            cv::cvtColor(src_opencv, dst_opencv, CV_YUV2BGR_IYUV);
            if(ncDst == 4) cv::cvtColor(dst_opencv, dst_opencv, cv::COLOR_BGR2BGRA);
            break;

        case YUV2GRAY_420:
            cv::cvtColor(src_opencv, dst_opencv, CV_YUV2GRAY_420);

            break;

        default:
            return HPC_NOT_SUPPORTED;
    }
    dst.toHost(output_data);
    checkResult_colortype<Tdst, ncDst>(output_data, dst_opencv.ptr<Tdst>(), output_height, output_width,
            dstep/sizeof(Tdst), dst_opencv.step/sizeof(Tdst), diff_THR, type);
    
    //x86
      checkResult_colortype<Tdst, ncDst>((Tdst*)dst_x86.ptr(), dst_opencv.ptr<Tdst>(), output_height, output_width,
            dstep/sizeof(Tdst), dst_opencv.step/sizeof(Tdst), diff_THR, type);
#elif defined(FASTCV_USE_OCL)
    Mat<Tdst, ncDst> dst(output_height, output_width, dstep);
    dst.fromHost(output_data);
#else
    Mat<Tdst, ncDst> dst(output_height, output_width, output_data, dstep);

    const Mat<Tsrc, ncSrc> src(input_height, input_width, input_data, step);

    HPCStatus_t sta = cvtColor<type, Tsrc, ncSrc, Tdst, ncDst>(src, &dst);
    EXPECT_EQ(HPC_SUCCESS, sta);

    cv::Mat dst_opencv(output_height, output_width, CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst), output_data_opencv, dstep);
    double time_fastcv;
    switch(type) {
         case BGR2I420:
            if(ncSrc == 4) cv::cvtColor(src_opencv, dst_opencv, CV_BGR2BGRA);
            cv::cvtColor(src_opencv, dst_opencv, CV_BGR2YUV_IYUV);
            break;

        case I4202BGR:
            cv::cvtColor(src_opencv, dst_opencv, CV_YUV2BGR_IYUV);
            if(ncDst == 4) cv::cvtColor(dst_opencv, dst_opencv, cv::COLOR_BGR2BGRA);
            break;

        case YUV2GRAY_420:
            cv::cvtColor(src_opencv, dst_opencv, CV_YUV2GRAY_420);
//printf("good\n");

            break;

        default:
            return HPC_NOT_SUPPORTED;
    }

    checkResult_colortype<Tdst, ncDst>((Tdst*)dst.ptr(), dst_opencv.ptr<Tdst>(), output_height, output_width,
            dstep/sizeof(Tdst), dst_opencv.step/sizeof(Tdst), diff_THR, type);
#endif



	free(output_data);
	free(output_data_opencv);
    free(input_data);
    free(input_data1);
    return 0;
}


TEST_F(I420cvtColorTest, I4202BGR){
    //ColorCvtType type = GetParam();
    imageParam p ;
  //  ASSERT_TRUE(!imageparams.empty());
 //   for(int i = 0; i < imageparams.size(); i++)
//    {
        p = imagep;
        testCvtColorFunc<uchar, 1, uchar, 3, I4202BGR, 1> (p,1.01);
        testCvtColorFunc<uchar, 1, uchar, 4, I4202BGR, 1> (p,1.01);
 //   }
  
}

TEST_F(I420cvtColorTest, BGR2I420){
    //ColorCvtType type = GetParam();
    imageParam p ;
  //  ASSERT_TRUE(!imageparams.empty());
   // for(int i = 0; i < imageparams.size(); i++)
 //   {
        p = imagep;
         testCvtColorFunc<uchar, 3, uchar, 1, BGR2I420, 1> (p,1.01);
         testCvtColorFunc<uchar, 4, uchar, 1, BGR2I420, 1> (p,1.01);
  //  }
  
}

TEST_F(I420cvtColorTest, YUV2GRAY_420){
    //ColorCvtType type = GetParam();
    imageParam p ;
  //  ASSERT_TRUE(!imageparams.empty());
   // for(int i = 0; i < imageparams.size(); i++)
  //  {
        p = imagep;
       testCvtColorFunc<uchar, 1, uchar, 1, YUV2GRAY_420, 1> (p,1.01);
  //  }
   
}




