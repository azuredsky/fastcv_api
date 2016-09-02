#include "test_head.hpp"
using namespace HPC::fastcv;
//#define PRINT_ALL

struct imageParam{
    size_t height;
    size_t width;
    size_t kernelx_len;
    size_t kernely_len;
    size_t spadding;
    size_t dpadding;
    bool allone;
};

 imageParam imagep;
 std::vector<imageParam> imageparams;

class erodeTest : public testing::Test
{
     public:
        static void SetUpTestCase()
        {
            std::vector<imageParam>::iterator iter = imageparams.begin();
            if(globalEnvironment::Instance()->testconfig.hasSection("erodeTest"))
            {
                Cfg::cfg_type vec_arg = globalEnvironment::Instance()->testconfig.sectionConfigVec("erodeTest");
                int num = vec_arg.size();
                for(int i = 0;i < num; i++)
                {
                    ASSERT_TRUE(vec_arg[i].find("height")!=vec_arg[i].end())<<"argument:height not found";
                    ASSERT_TRUE(vec_arg[i].find("width")!=vec_arg[i].end())<<"argument:width not found";
                    ASSERT_TRUE(vec_arg[i].find("kernelx_len")!=vec_arg[i].end())<<"argument:kernelx_len not found";
                    ASSERT_TRUE(vec_arg[i].find("kernely_len")!=vec_arg[i].end())<<"argument:kernely_len not found";                   
                    ASSERT_TRUE(vec_arg[i].find("spadding")!=vec_arg[i].end())<<"argument:spadding not found";
                    ASSERT_TRUE(vec_arg[i].find("dpadding")!=vec_arg[i].end())<<"argument:dpadding not found";
                    ASSERT_TRUE(vec_arg[i].find("allone")!=vec_arg[i].end())<<"argument:allone not found";
                    imagep.height = std::stoul(vec_arg[i]["height"]);
                    imagep.width = std::stoul(vec_arg[i]["width"]);
                    imagep.kernelx_len = std::stoul(vec_arg[i]["kernelx_len"]);
                    imagep.kernely_len = std::stoul(vec_arg[i]["kernely_len"]);              
                    imagep.spadding = std::stoi(vec_arg[i]["spadding"]);
                    imagep.dpadding = std::stoi(vec_arg[i]["dpadding"]);
                    imagep.allone = (std::stoi(vec_arg[i]["allone"])!=0);
                    imageparams.push_back(imagep);

                }
            }
            else
            {
                imagep={257, 189, 12, 12, 3, 3, true};
                imageparams.push_back(imagep);
            }

        }
        static void TearDownTestCase()
        {
            if(!imageparams.empty())
                imageparams.clear();
        }
   // public:
//        imageParam imagep;
  //      std::vector<imageParam> imageparams;
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
void checkResult(const T *data1, const T *data2, const int height, const int width, const float diff_THR) {
    double max = DBL_MIN;
    double min = DBL_MAX;
    double temp_diff = 0.0;

    int num = height*width *nc;
    for(int i=0; i < num; i++){
        //ASSERT_NEAR(data1[i], data2[i], 1e-5);
        temp_diff = fabs((double) *(data1 + i)-(double) *(data2 + i));

#ifdef PRINT_ALL
		if(temp_diff > 1e-5)
            std::cout << "Print data12: " << i << " : " << (float) data1[i] << " : " << (float) data2[i] << " : " << temp_diff << std::endl;
#endif
        max = (temp_diff > max) ? temp_diff : max;
        min = (temp_diff < min) ? temp_diff : min;
    }
    EXPECT_LT(max, diff_THR);
    //ASSERT_NEAR(order, 1e-5, 1e-5);
}

template<typename T, int nc>
void checkResult(const T *data1, const T *data2, const int height, const int width, int dstep, const float diff_THR) {
    double max = DBL_MIN;
    double min = DBL_MAX;
    double temp_diff = 0.0;

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width*nc; j++){
            float val1 = data1[i * dstep +j];
            float val2 = data2[i * dstep + j];

            temp_diff = fabs(val1 - val2);

#ifdef PRINT_ALL
            int flag = 1;
            if(temp_diff > diff_THR && flag == 1){
                std::cout << "Print data12: " << i << " : " <<  j << " : "<< (float) val1 << " : " << (float) val2 << " : " << temp_diff << std::endl;
               flag = 0;
            }
#endif
            max = (temp_diff > max) ? temp_diff : max;
            min = (temp_diff < min) ? temp_diff : min;
        }
    }
    EXPECT_LT(max, diff_THR);
}


/*
int main(int argc, char *argv[]){
    if(argc != 1 && argc != 8){
        printf("Error params!\n");
        printf("Usage: ./run height width kernelx_len kernely_len spadding dpadding allone\n");
        return 0;
    }
   // imagep={64, 64, 64, 64, 3, 3, true};
    imagep={257, 189, 12, 12, 3, 3, true};
    if(argc == 8){
        imagep.height = atoi(argv[1]);
        imagep.width = atoi(argv[2]);
        imagep.kernelx_len = atoi(argv[3]);
        imagep.kernely_len = atoi(argv[4]);
        imagep.spadding = atoi(argv[5]);
        imagep.dpadding = atoi(argv[6]);
        imagep.allone = (atoi(argv[7])!=0);
    }

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}*/

template<typename T, int ncSrc, int ncDst, int nc, int val_div>
int testErodeFunc(imageParam imagep,float diff_THR) {
    size_t height = imagep.height;
    size_t width = imagep.width;
    size_t kernelx_len = imagep.kernelx_len;
    size_t kernely_len = imagep.kernely_len;
    int spadding = imagep.spadding;
    int dpadding = imagep.dpadding;
    bool allone = imagep.allone;

    size_t input_channels = nc;
    size_t output_channels = nc;

    printf("Input Mat scale(= Output Mat scale): width*height*channels = %ld * %ld * %ld\n", width, height, input_channels);
    size_t step = (width * input_channels + spadding)*sizeof(T);
    T* input_data = (T*)malloc(height*step);
    randomPaddingData<T, val_div>(input_data, height, width*input_channels, width * input_channels + spadding);

    uchar* input_element = (uchar*)malloc(kernelx_len * kernely_len);
    if (allone){
        for (size_t i = 0; i < kernelx_len * kernely_len; ++i) input_element[i] = 1;
    }
    else{
        randomPaddingData<uchar, 128>(input_element, kernelx_len, kernely_len, kernely_len);
    }


    struct timeval begin, stop;


#if defined(FASTCV_USE_X86) && !defined(FASTCV_USE_CUDA)
    const Mat<T, nc> src(height, width, input_data, step);
    const Mat<uchar, 1> element(kernelx_len, kernely_len, input_element, kernely_len * sizeof(uchar));

    size_t dstep = (width * output_channels + dpadding) * sizeof(T);
    T* output_data = (T*) malloc(height * dstep);
    T* output_data_opencv = (T*)malloc(height*dstep);
    randomPaddingData<T, val_div>(output_data, height,
        width * output_channels, width * output_channels + dpadding);

    randomPaddingData<T, val_div>(output_data_opencv, height, 
        width * output_channels, width * output_channels + dpadding);

    Mat<T, ncDst> dst(height, width, output_data, dstep);

    cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), input_data, step);
    cv::Mat element_opencv(kernelx_len, kernely_len, CV_MAKETYPE(cv::DataType<uchar>::depth, 1), input_element, kernely_len * sizeof(uchar));

    gettimeofday(&begin,NULL);

    HPCStatus_t sta = erode<T, ncSrc, ncDst, nc, EcoEnv_t::ECO_ENV_X86>(src, element, &dst);
    gettimeofday(&stop,NULL);
    EXPECT_EQ(HPC_SUCCESS, sta);

    cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), output_data_opencv, dstep);
    cv::erode(src_opencv, dst_opencv, element_opencv);

    checkResult<T, nc>((T*)dst.ptr(), dst_opencv.ptr<T>(), height, width, dstep/sizeof(T), diff_THR);
#endif

#if defined(FASTCV_USE_ARM)
    const Mat<T, nc> src(height, width, input_data, step);
    const Mat<uchar, 1> element(kernelx_len, kernely_len, input_element, kernely_len * sizeof(uchar));

    size_t dstep = (width * output_channels + dpadding)*sizeof(T);
    T* output_data = (T*)malloc(height*dstep);
    T* output_data_opencv = (T*)malloc(height*dstep);
    randomPaddingData<T, val_div>(output_data, height,
            width*output_channels, width * output_channels + dpadding);

    randomPaddingData<T, val_div>(output_data_opencv, height,
            width*output_channels, width * output_channels + dpadding);

    Mat<T, ncDst> dst(height, width, output_data, dstep);

    cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), input_data, step);
    cv::Mat element_opencv(kernelx_len, kernely_len, CV_MAKETYPE(cv::DataType<uchar>::depth, 1), input_element, kernely_len * sizeof(uchar));

	gettimeofday(&begin,NULL);

    HPCStatus_t sta = erode<T, ncSrc, ncDst, nc, EcoEnv_t::ECO_ENV_ARM>(src, element, &dst);

	gettimeofday(&stop,NULL);
    EXPECT_EQ(HPC_SUCCESS, sta);

    cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), output_data_opencv, dstep);
    cv::erode(src_opencv, dst_opencv, element_opencv);

    checkResult<T, nc>((T*)dst.ptr(), dst_opencv.ptr<T>(), height, width, dstep/sizeof(T), diff_THR);
#endif

    return 0;
}

TEST_F(erodeTest, uchar_uchar){
    imageParam p;
  //  ASSERT_TRUE(!imageparams.empty());
  //  for(int i = 0;i < imageparams.size();i++)
   // {
        p = imagep;
        testErodeFunc<uchar, 1, 1, 1, 1>(p,1e-5);
        testErodeFunc<uchar, 3, 3, 3, 1>(p,1e-5);
        testErodeFunc<uchar, 4, 4, 4, 1>(p,1e-5);
   // }
   
}


TEST_F(erodeTest, float_float){
    imageParam p;
  //  ASSERT_TRUE(!imageparams.empty());
   // for(int i = 0;i < imageparams.size();i++)
    //{
        p = imagep;
        testErodeFunc<float, 1, 1, 1, 1>(p,1e-5);
        testErodeFunc<float, 3, 3, 3, 1>(p,1e-5);
        testErodeFunc<float, 4, 4, 4, 1>(p,1e-5);
   // }

}
