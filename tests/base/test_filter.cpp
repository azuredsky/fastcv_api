#include "test_head.hpp"
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


class filterTest : public testing::Test
{
     public:
        virtual void SetUp()
        {
            std::vector<imageParam>::iterator iter = imageparams.begin();
            if(globalEnvironment::Instance()->testconfig.hasSection("filterTest"))
            {
                Cfg::cfg_type vec_arg = globalEnvironment::Instance()->testconfig.sectionConfigVec("filterTest");
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
                imagep={10, 10, 10, 10, 7, 11};
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
        data[i] = (T)( (float)tmp/(float)val );
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


template<typename T>
void randomKernel(T *data, const size_t kernelSize){
    size_t tmp;
    clock_t ct = clock();
    srand((unsigned int)ct);

    T sum_val = 0;
    for(size_t i = 0; i < kernelSize * kernelSize; i++){
        tmp = rand() % 100;
        sum_val += (T)( (float)tmp/100.f );
        data[i] = (T)( (float)tmp/100.f );
    }
    for(size_t i = 0; i < kernelSize * kernelSize; i++){
        data[i] = (T)( (float)data[i]/(float)sum_val );
    }
}

template<typename T, int nc>
void checkResult(const T *data1, const T *data2, const int height, const int width, const float diff_THR) {
    double max = DBL_MIN;
    double min = DBL_MAX;
    double temp_diff = 0.0;
    //FILE* fp;
    //fp = fopen("Filterdiff","w+");

    int num = height*width *nc;
    for(int i=0; i < num; i++){
        //ASSERT_NEAR(data1[i], data2[i], 1e-5);
        temp_diff = fabs((double) *(data1 + i)-(double) *(data2 + i));
        //if(temp_diff>0)
        //fprintf(fp,"%d me %f opencv %f  diff %f\n",i,(float)data1[i],(float)data2[i],temp_diff);
#ifdef PRINT_ALL
        std::cout << "Print data12: " << i << " : " << (float) data1[i] << " : " << (float) data2[i] << " : " << temp_diff << std::endl;
#endif
        max = (temp_diff > max) ? temp_diff : max;
        min = (temp_diff < min) ? temp_diff : min;
    }
    EXPECT_LT(max, diff_THR);
    //ASSERT_NEAR(order, 1e-5, 1e-5);
    //fclose(fp);
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
    imagep={10, 10, 10, 10, 7, 11};
    imagep={1920, 1080, 1920, 1080, 0, 0};
   		imagep={640, 1080, 640, 1080};
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
template<typename T, int filterSize, int ncSrc, int ncDst, int nc, BorderType bt, int val_div>
int testFilterFunc(imageParam imagep,float diff_THR) {
    size_t input_width = imagep.srcWidth;
    size_t input_height = imagep.srcHeight;
    size_t input_channels = nc;
    size_t output_width = imagep.dstWidth;
    size_t output_height = imagep.dstHeight;
    size_t output_channels = nc;
    int spadding = imagep.spadding;
    int dpadding = imagep.dpadding;

    printf("Input Mat scale: width*height*channels = %ld*%ld*%ld\n", input_width, input_height, input_channels);
    printf("Output Mat scale: width*height*channels = %ld*%ld*%ld\n", output_width, output_height, output_channels);

   /* size_t input_data_num = input_width*input_height*input_channels;

    T* input_data = (T*)malloc(input_data_num*sizeof(T));

    randomRangeData<T, val_div>(input_data, input_data_num);*/

    size_t step = (input_width * input_channels + spadding)*sizeof(T);
    T* input_data = (T*)malloc(input_height*step);
    randomPaddingData<T, val_div>(input_data, input_height, input_width*input_channels, input_width * input_channels + spadding);

   /* std::cout << " Original data:" << std::endl;
    for(size_t i = 0; i < input_data_num; i++)
        std::cout << i << " : " << (float) input_data[i] << " " << std::endl;*/

    struct timeval start, end;

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
    MatX2D<float, filterSize, filterSize> f;
    randomKernel<float>(f.ptr(), (size_t) filterSize);

    cv::Mat f_opencv(filterSize, filterSize, CV_32FC1, f.ptr());
    // call opencv to verify result
    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<T>::depth, ncSrc), input_data, step);
    cv::Mat dst_opencv(output_height, output_width,
            CV_MAKETYPE(cv::DataType<T>::depth, ncDst), output_data_opencv, dstep);
    //warm up
    cv::filter2D(src_opencv, dst_opencv, -1, f_opencv);
    gettimeofday(&start, NULL);
    cv::filter2D(src_opencv, dst_opencv, -1, f_opencv);
    gettimeofday(&end, NULL);
    double time_opencv = timingExec(start, end);

    // call fastcv
    //warm up
    HPCStatus_t sta = filter2D<bt, T, filterSize, ncSrc, ncDst, nc>(src, &dst, f);
    gettimeofday(&start, NULL);
    sta = filter2D<bt, T, filterSize, ncSrc, ncDst, nc>(src, &dst, f);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    double time_fastcv = timingExec(start, end);
    EXPECT_EQ(HPC_SUCCESS, sta);

    //x86
    HPCStatus_t sta_x86 = filter2D<bt, T, filterSize, ncSrc, ncDst, nc>(src_x86, &dst_x86, f);
    gettimeofday(&start, NULL);
    sta_x86 = filter2D<bt, T, filterSize, ncSrc, ncDst, nc>(src_x86, &dst_x86, f);
    gettimeofday(&end, NULL);
    double time_fastcv_x86 = timingExec(start, end);
    EXPECT_EQ(HPC_SUCCESS, sta_x86);

    printf("fastcv_cuda cost %f   fastcv_x86 cost %f   opencv cost %f\n", time_fastcv, time_fastcv_x86,time_opencv);
    // check results

    //Mat<T, ncSrc> why(input_height, input_width, output_data1, dstep);
    //why.deepCopy(dst);
#elif defined(FASTCV_USE_OCL)

   MatX2D<float, filterSize, filterSize> f;
    randomKernel<float>(f.ptr(), (size_t) filterSize);

    cv::Mat f_opencv(filterSize, filterSize, CV_32FC1, f.ptr());
    // call opencv to verify result
    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<T>::depth, ncSrc), input_data, step);
    cv::Mat dst_opencv(output_height, output_width,
            CV_MAKETYPE(cv::DataType<T>::depth, ncDst), output_data_opencv, dstep);
    //warm up
    cv::filter2D(src_opencv, dst_opencv, -1, f_opencv);
    gettimeofday(&start, NULL);
    cv::filter2D(src_opencv, dst_opencv, -1, f_opencv);
    gettimeofday(&end, NULL);
    double time_opencv = timingExec(start, end);

    // call fastcv
    //warm up
    HPCStatus_t sta = filter2D<bt, T, filterSize, ncSrc, ncDst, nc>(src, &dst, f);
    gettimeofday(&start, NULL);
    sta = filter2D<bt, T, filterSize, ncSrc, ncDst, nc>(src, &dst, f);
    gettimeofday(&end, NULL);
    double time_fastcv = timingExec(start, end);
    EXPECT_EQ(HPC_SUCCESS, sta);

    printf("fastcv cost %f    opencv cost %f\n", time_fastcv, time_opencv);
    // check results

    //Mat<T, ncSrc> why(input_height, input_width, output_data1, dstep);
    //why.deepCopy(dst);


#else
    MatX2D<float, filterSize, filterSize> f;
    randomKernel<float>(f.ptr(), (size_t) filterSize);

    cv::Mat f_opencv(filterSize, filterSize, CV_32FC1, f.ptr());
    // call opencv to verify result
    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<T>::depth, ncSrc), input_data, step);
    cv::Mat dst_opencv(output_height, output_width,
            CV_MAKETYPE(cv::DataType<T>::depth, ncDst), output_data_opencv, dstep);
    //warm up
    cv::filter2D(src_opencv, dst_opencv, -1, f_opencv);
    gettimeofday(&start, NULL);
    cv::filter2D(src_opencv, dst_opencv, -1, f_opencv);
    gettimeofday(&end, NULL);
    double time_opencv = timingExec(start, end);

    // call fastcv
    //warm up
    HPCStatus_t sta = filter2D<bt, T, filterSize, ncSrc, ncDst, nc>(src, &dst, f);
    gettimeofday(&start, NULL);
    sta = filter2D<bt, T, filterSize, ncSrc, ncDst, nc>(src, &dst, f);
    gettimeofday(&end, NULL);
    double time_fastcv = timingExec(start, end);
    EXPECT_EQ(HPC_SUCCESS, sta);

    printf("fastcv cost %f    opencv cost %f\n", time_fastcv, time_opencv);
    // check results

    Mat<T, ncSrc> why(input_height, input_width, output_data1, dstep);
    why.deepCopy(dst);
#endif

#if defined(FASTCV_USE_CUDA)
    memset(output_data,0,output_height * dstep);
    dst.toHost(output_data);
    checkResult<T, ncDst>(output_data, dst_opencv.ptr<T>(), output_height,
            output_width, dstep/sizeof(T), diff_THR);
   
   //x86
    checkResult<T, ncDst>(output_data, dst_opencv.ptr<T>(), output_height, output_width, dstep/sizeof(T), diff_THR);

#elif defined(FASTCV_USE_OCL)
    dst.toHost(output_data);
    checkResult<T, ncDst>(output_data, dst_opencv.ptr<T>(), output_height, output_width, dstep/sizeof(T), diff_THR);
#else
    checkResult<T, ncDst>(output_data, dst_opencv.ptr<T>(), output_height, output_width, dstep/sizeof(T), diff_THR);
#endif

    return 0;
}

TEST_F(filterTest, uchar){
    imageParam p;
  //  ASSERT_TRUE(!imageparams.empty());
  //  for(int i = 0; i < imageparams.size(); i++)
 //   {
        p = imagep;
        testFilterFunc<uchar, 3, 1, 1, 1, BORDER_DEFAULT, 1>(p,1.1);
        testFilterFunc<uchar, 3, 3, 3, 3, BORDER_DEFAULT, 1>(p,1.1);
        testFilterFunc<uchar, 3, 4, 4, 4, BORDER_DEFAULT, 1>(p,1.1);
 //   }
    
}
TEST_F(filterTest, float_1){
    imageParam p;
  //  ASSERT_TRUE(!imageparams.empty());
 //   for(int i = 0; i<imageparams.size(); i++)
 //   {
        p = imagep;
        testFilterFunc<float, 3, 1, 1, 1, BORDER_DEFAULT, 255>(p,1e-5);
        testFilterFunc<float, 3, 3, 3, 3, BORDER_DEFAULT, 255>(p,1e-5);
        testFilterFunc<float, 3, 4, 4, 4, BORDER_DEFAULT, 255>(p,1e-5);
 //   }
    
}


