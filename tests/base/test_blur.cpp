#include "test_head.hpp" 
using namespace HPC::fastcv;
//#define PRINT_ALL
struct imageParam{
    size_t srcWidth;
    size_t srcHeight;
    size_t dstWidth;
    size_t dstHeight;
    size_t kernelx_len;
    size_t kernely_len;
    bool normalize;
    int spadding;
    int dpadding;
};

class BlurTest : public testing::Test
{
    public:
        virtual void SetUp()
        {
            std::vector<imageParam>::iterator iter = imageparams.begin();
            if(globalEnvironment::Instance()->testconfig.hasSection("BlurTest"))
            {
                Cfg::cfg_type vec_arg = globalEnvironment::Instance()->testconfig.sectionConfigVec("BlurTest");
                int num = vec_arg.size();
                for(int i = 0;i < num; i++)
                {
                    ASSERT_TRUE(vec_arg[i].find("srcWidth")!=vec_arg[i].end())<<"argument:srcWidth not found";
                    ASSERT_TRUE(vec_arg[i].find("srcHeight")!=vec_arg[i].end())<<"argument:srcHeight not found";
                    ASSERT_TRUE(vec_arg[i].find("dstWidth")!=vec_arg[i].end())<<"argument:dstWidth not found";
                    ASSERT_TRUE(vec_arg[i].find("dstHeight")!=vec_arg[i].end())<<"argument:dstHeight not found";
                    ASSERT_TRUE(vec_arg[i].find("kernelx_len")!=vec_arg[i].end())<<"argument:kernelx_len not found";
                    ASSERT_TRUE(vec_arg[i].find("kernely_len")!=vec_arg[i].end())<<"argument:kernely_len not found";
                    ASSERT_TRUE(vec_arg[i].find("normalize")!=vec_arg[i].end())<<"argument:normalize not found";
                    ASSERT_TRUE(vec_arg[i].find("spadding")!=vec_arg[i].end())<<"argument:spadding not found";
                    ASSERT_TRUE(vec_arg[i].find("dpadding")!=vec_arg[i].end())<<"argument:dpadding not found";
                    imagep.srcWidth = std::stoul(vec_arg[i]["srcWidth"]);
                    imagep.srcHeight = std::stoul(vec_arg[i]["srcHeight"]);
                    imagep.dstWidth = std::stoul(vec_arg[i]["dstWidth"]);
                    imagep.dstHeight = std::stoul(vec_arg[i]["dstHeight"]);
                    imagep.kernelx_len = std::stoul(vec_arg[i]["kernelx_len"]);
                    imagep.spadding = std::stoi(vec_arg[i]["spadding"]);
                    imagep.dpadding = std::stoi(vec_arg[i]["dpadding"]);
                    imagep.normalize = (std::stoi(vec_arg[i]["normalize"])!=0);
                  //   iter = imageparams.insert(iter,imagep);

                }
            }
            else
            {
                imagep={970, 970, 970, 970, 3, 3, true, 7, 11};
              //  imageparams.insert(iter,imagep);
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
void checkResult(const T *data1, const T *data2, const int height, const int width, const float diff_THR) {
    double max = DBL_MIN;
    double min = DBL_MAX;
    double temp_diff = 0.0;

    int num = height*width *nc;
    for(int i=0; i < num; i++){
        //ASSERT_NEAR(data1[i], data2[i], 1e-5);
        temp_diff = fabs((double) *(data1 + i)-(double) *(data2 + i));

#ifdef PRINT_ALL
//		if(temp_diff > 1e-5)
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

    int num = height*width *nc;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width*nc; j++){
            float val1 = data1[i * dstep +j];
            float val2 = data2[i * dstep + j];

            temp_diff = fabs(val1 - val2);

#ifdef PRINT_ALL
            int flag = 1;
           // if(temp_diff > diff_THR && flag == 1){
                std::cout << "Print data12: " << i << " : " <<  j << " : "<< (float) val1 << " : " << (float) val2 << " : " << temp_diff << std::endl;
               flag = 0;
            //}
#endif
            max = (temp_diff > max) ? temp_diff : max;
            min = (temp_diff < min) ? temp_diff : min;
        }
    }
    EXPECT_LT(max, diff_THR);
}

/*
int main(int argc, char *argv[]){
    if(argc != 1 && argc != 10){
        printf("Error params!\n");
        printf("Usage: ./run src_width src_height \
                dst_width dst_height \n");
    }
   // imagep={64, 64, 64, 64, 3, 3, true};
    imagep={970, 970, 970, 970, 3, 3, true, 7, 11};
    imagep={2023, 3034, 2023, 3034, 3, 3, true, 7, 11};
    imagep={2023, 3034, 2023, 3034, 3, 3, true, 0,0};
    if(argc == 10){
        imagep.srcWidth = atoi(argv[1]);
        imagep.srcHeight = atoi(argv[2]);
        imagep.dstWidth = atoi(argv[3]);
        imagep.dstHeight = atoi(argv[4]);
        imagep.kernelx_len = atoi(argv[5]);
        imagep.kernely_len = atoi(argv[6]);
        imagep.normalize = (atoi(argv[7])!=0);
        imagep.spadding = atoi(argv[8]);
        imagep.dpadding = atoi(argv[9]);
    }

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
*/
template<typename T, int ncSrc, int ncDst, int nc, BorderType bt, int val_div>
int testboxFilterFunc(imageParam imagep,float diff_THR) {
    size_t input_width = imagep.srcWidth;
    size_t input_height = imagep.srcHeight;
    size_t input_channels = nc;
    size_t output_width = imagep.dstWidth;
    size_t output_height = imagep.dstHeight;
    size_t output_channels = nc;
    int kernelx_len = imagep.kernelx_len;
    int kernely_len = imagep.kernely_len;
    bool normalized = imagep.normalize;
    int spadding = imagep.spadding;
    int dpadding = imagep.dpadding;

    printf("Input Mat scale: width*height*channels = %ld * %ld * %ld\n", input_width, input_height, input_channels);
    printf("Output Mat scale: width*height*channels = %ld * %ld * %ld\n", output_width, output_height, output_channels);

    size_t step = (input_width * input_channels + spadding)*sizeof(T);
    T* input_data = (T*)malloc(input_height*step);
    randomPaddingData<T, val_div>(input_data, input_height, input_width*input_channels, input_width * input_channels + spadding);

   /* for(int i = 0; i < input_height;i++){
        for(int j = 0; j < output_width; j++){
            std::cout<<(int)input_data[i*step + j]<<" ";   
        }
        std::cout<<std::endl;
    }*/

    struct timeval begin, stop;
	double time_me = 0;
#if defined(FASTCV_USE_CUDA)
    Mat<T, nc> src(input_height, input_width, step);
    src.fromHost(input_data);

    const Mat<T, nc,EcoEnv_t::ECO_ENV_X86> src_x86(input_height, input_width, input_data, step);

#elif defined(FASTCV_USE_OCL)
    Mat<T, nc> src(input_height, input_width, step);
    src.fromHost(input_data);
#else
    const Mat<T, nc> src(input_height, input_width, input_data, step);
#endif


    size_t dstep = (output_width * output_channels + dpadding)*sizeof(T);
    T* output_data = (T*)malloc(output_height*dstep);
    T* output_data1 = (T*)malloc(output_height*dstep);
    T* output_data_opencv = (T*)malloc(output_height*dstep);
    randomPaddingData<T, val_div>(output_data, output_height,
            output_width*output_channels, output_width * output_channels + dpadding);

    randomPaddingData<T, val_div>(output_data_opencv, output_height,
            output_width*output_channels, output_width * output_channels + dpadding);

#if defined(FASTCV_USE_CUDA)
    Mat<T, ncDst> dst(output_height, output_width, dstep);
    dst.fromHost(output_data);

    Mat<T, ncDst,EcoEnv_t::ECO_ENV_X86> dst_x86(output_height, output_width, output_data, dstep);

#elif defined(FASTCV_USE_OCL)
    Mat<T, ncDst> dst(output_height, output_width, dstep);
    dst.fromHost(output_data);
#else
    Mat<T, ncDst> dst(output_height, output_width, output_data, dstep);
#endif



#if defined(FASTCV_USE_CUDA)
    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<T>::depth, nc), input_data, step);

    Size2i ks(kernelx_len, kernely_len);
    Point2i arch(-1, -1);

    //cuda
	gettimeofday(&begin,NULL);
    HPCStatus_t sta = boxFilter<bt, T, ncSrc, ncDst, nc>(src, ks, arch, &dst, normalized);
	gettimeofday(&stop,NULL);

	time_me = timingExec(begin,stop);
    EXPECT_EQ(HPC_SUCCESS, sta);

    //x86
    gettimeofday(&begin,NULL);
    HPCStatus_t sta_x86 = boxFilter<bt, T, ncSrc, ncDst, nc>(src_x86, ks, arch, &dst_x86, normalized);
	gettimeofday(&stop,NULL);

	time_me = timingExec(begin,stop);
    EXPECT_EQ(HPC_SUCCESS, sta_x86);


    cv::Mat dst_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<T>::depth, nc), output_data_opencv, dstep);
    cv::Size ksize(kernelx_len, kernely_len);
    cv::boxFilter(src_opencv, dst_opencv, -1, ksize, cv::Point(-1,-1), normalized);

#elif defined(FASTCV_USE_OCL)
    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<T>::depth, nc), input_data, step);

    Size2i ks(kernelx_len, kernely_len);
    Point2i arch(-1, -1);


	gettimeofday(&begin,NULL);
    HPCStatus_t sta = boxFilter<bt, T, ncSrc, ncDst, nc>(src, ks, arch, &dst, normalized);
	gettimeofday(&stop,NULL);

	time_me = timingExec(begin,stop);
    EXPECT_EQ(HPC_SUCCESS, sta);

    cv::Mat dst_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<T>::depth, nc), output_data_opencv, dstep);
    cv::Size ksize(kernelx_len, kernely_len);
    cv::boxFilter(src_opencv, dst_opencv, -1, ksize, cv::Point(-1,-1), normalized);


#else
    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<T>::depth, nc), input_data, step);

    Size2i ks(kernelx_len, kernely_len);
    Point2i arch(-1, -1);
	gettimeofday(&begin,NULL);
    HPCStatus_t sta = boxFilter<bt, T, ncSrc, ncDst, nc>(src, ks, arch, &dst, normalized);
	gettimeofday(&stop,NULL);

	time_me = timingExec(begin,stop);
    printf("time of factcv: %2f\n", time_me);
    EXPECT_EQ(HPC_SUCCESS, sta);

    cv::Mat dst_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<T>::depth, nc), output_data_opencv, dstep);
    cv::Size ksize(kernelx_len, kernely_len);
	
    gettimeofday(&begin,NULL);
    cv::boxFilter(src_opencv, dst_opencv, -1, ksize, cv::Point(-1,-1), normalized);
	gettimeofday(&stop,NULL);
	time_me = timingExec(begin,stop);
    printf("time of opencv: %2f\n", time_me);
#endif


#if defined(FASTCV_USE_CUDA)
    dst.toHost(output_data);
    checkResult<T, nc>(output_data, dst_opencv.ptr<T>(), input_height, input_width, dstep/sizeof(T), diff_THR);

    //check x86 result
    checkResult<T, nc>((T*)dst_x86.ptr(), dst_opencv.ptr<T>(), input_height, input_width, dstep/sizeof(T), diff_THR);

#elif defined(FASTCV_USE_OCL)
    dst.toHost(output_data);
    checkResult<T, nc>(output_data, dst_opencv.ptr<T>(), input_height, input_width, dstep/sizeof(T), diff_THR);
#else
    checkResult<T, nc>((T*)dst.ptr(), dst_opencv.ptr<T>(), input_height, input_width, dstep/sizeof(T), diff_THR);
#endif

    return 0;
}
TEST_F(BlurTest, float_float){
    imageParam p ;
    ASSERT_TRUE(!imageparams.empty());
    for(int i = 0; i < imageparams.size();i++)
    {
        p = imageparams[i];
        testboxFilterFunc<float, 1, 1, 1, BORDER_DEFAULT, 255>(p,1e-5);
        testboxFilterFunc<float, 3, 3, 3, BORDER_DEFAULT, 255>(p,1e-5);
        testboxFilterFunc<float, 4, 4, 4, BORDER_DEFAULT, 255>(p,1e-5);

    }
   
}
TEST_F(BlurTest, uchar_uchar){
    imageParam p;
//    ASSERT_TRUE(!imageparams.empty());
 //   for(int i = 0; i < imageparams.size(); i++)
//    {
        p = imagep;
        testboxFilterFunc<uchar, 1, 1, 1, BORDER_DEFAULT, 1>(p,1e-5);
        testboxFilterFunc<uchar, 3, 3, 3, BORDER_DEFAULT, 1>(p,1e-5);
        testboxFilterFunc<uchar, 4, 4, 4, BORDER_DEFAULT, 1>(p,1e-5);
  //  }
    
}
