#include "test_head.hpp"
#include <opencv2/gpu/gpu.hpp>
#include<opencv2/ocl/ocl.hpp>


using namespace std;
#define N 1000
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


class gaussianBlurTest : public testing::Test
{
    public:
        virtual void SetUp()
        {
            std::vector<imageParam>::iterator iter = imageparams.begin();
            if(globalEnvironment::Instance()->testconfig.hasSection("gaussianBlurTest"))
            {
                Cfg::cfg_type vec_arg = globalEnvironment::Instance()->testconfig.sectionConfigVec("gaussianBlurTest");
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
                imagep={12, 12, 12,12,7,11};
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
                //printf("%d\n",data[i]);
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

        int dstep = width*nc;
        //int num = height*width *nc;
        for(int i = 0; i < height; i++ ){
            for(int j = 0; j < width * nc; j++ ){
                //ASSERT_NEAR(data1[i], data2[i], 1e-5);
               // temp_diff = fabs((double) *(data1 + i)-(double) *(data2 + i));
                temp_diff = data1[i * dstep + j] - data2[i * dstep + j];
#ifdef PRINT_ALL
                //  if(temp_diff > diff_THR)
                std::cout << "Print data12: " << i << " : " << j << " : " << (float) data1[i * dstep + j] << " : " << (float) data2[i * dstep  + j] << " : " << temp_diff << std::endl;
#endif
                max = (temp_diff > max) ? temp_diff : max;
                min = (temp_diff < min) ? temp_diff : min;
            }
        }
        EXPECT_LT(max, diff_THR);
        //ASSERT_NEAR(order, 1e-5, 1e-5);
}


template<typename T, int nc>
void checkResult(const T *data1, const T *data2, const int height, const int width, int dstep, const float diff_THR) {
    double max = DBL_MIN;
    double min = DBL_MAX;
    double temp_diff = 0.0;

    //int num = height*width *nc;
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






template<typename Tsrc,int ncSrc, typename Tdst ,int ncDst, int nc, int filterSize, int val_div>
int testGaussianBlurFunc(imageParam p,float diff_THR) {
    size_t input_width = p.srcWidth;
    size_t input_height = p.srcHeight;
    size_t input_channels = ncSrc;
    size_t output_width = p.dstWidth;
    size_t output_height = p.dstHeight;
    size_t output_channels = ncDst;
    int spadding = p.spadding;
    int dpadding = p.dpadding;

    printf("Input Mat scale: width*height*channels = %ld*%ld*%ld\n", input_width, input_height, input_channels);
    printf("Output Mat scale: width*height*channels = %ld*%ld*%ld\n", output_width, output_height, output_channels);

    //size_t input_data_num = input_width*input_height*input_channels;

    size_t step = (input_width * input_channels + spadding)*sizeof(Tsrc);
    Tsrc* input_data = (Tsrc*)malloc(input_height*step);
    randomPaddingData<Tsrc, val_div>(input_data, input_height,
            input_width*input_channels, input_width * input_channels + spadding);
    //randomRangeData<Tsrc,val_div>(input_data, input_data_num);
/*    for(int i=0;i<input_data_num;i++){
        fprintf(fp,"data : %d  %d\n",i,input_data[i]);
    }*/

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


#if defined(FASTCV_USE_OCL)
    struct timeval start, end;
    // call opencv to verify result
    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, ncSrc), input_data, step);
    cv::Mat dst_opencv(output_height, output_width,
            CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst), output_data_opencv, dstep);
    cv::ocl::oclMat src_opencvD(src_opencv);
    cv::ocl::oclMat dst_opencvD;

    cv::Size ksize;
    ksize.height = filterSize;
    ksize.width  = filterSize;

    //warm up
    cv::ocl::GaussianBlur(src_opencvD,dst_opencvD,ksize,-1,-1);
    gettimeofday(&start, NULL);
    for(int i =0;i<N;i++){
        cv::ocl::GaussianBlur(src_opencvD,dst_opencvD,ksize,-1,-1);
        cv::ocl::finish();
    }
    gettimeofday(&end, NULL);
    double time_opencv = timingExec(start, end);
    dst_opencvD.download(dst_opencv);

    //Mat<Tdst, ncDst> dst(output_height, output_width);
    //warm up
    HPCStatus_t sta = gaussianBlur<Tsrc,filterSize,ncSrc, ncDst, nc>(src, &dst);
     sta = gaussianBlur<Tsrc,filterSize,ncSrc, ncDst, nc>(src, &dst);
    gettimeofday(&start, NULL);
    for(int i =0;i<N;i++){
        sta = gaussianBlur<Tsrc,filterSize,ncSrc, ncDst, nc>(src, &dst);
        clFinish(opencl.getCommandQueue());
    }
    gettimeofday(&end, NULL);
    EXPECT_EQ(HPC_SUCCESS, sta);

    double time_fastcv = timingExec(start, end);

    printf("fastcv cost %f  opencv cost %f\n", time_fastcv, time_opencv);

#elif defined(FASTCV_USE_CUDA)
    struct timeval start, end;
    // call opencv to verify result
    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, ncSrc), input_data, step);
    cv::Mat dst_opencv(output_height, output_width,
            CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst), output_data_opencv, dstep);
    cv::gpu::GpuMat src_opencvD(src_opencv);
    cv::gpu::GpuMat dst_opencvD;

    cv::Size ksize;
    ksize.height = filterSize;
    ksize.width  = filterSize;

    cout<<"before gaussianBlur dst_opencv step  "<<  dst_opencv.step <<endl;
    cout<<"dstep "<<dstep <<endl;
    //warm up
    cv::gpu::GaussianBlur(src_opencvD,dst_opencvD,ksize,-1,-1);
    gettimeofday(&start, NULL);
    for(int i =0;i<N;i++){
        cv::gpu::GaussianBlur(src_opencvD,dst_opencvD,ksize,-1,-1);
        cudaDeviceSynchronize();
    }
    gettimeofday(&end, NULL);

    double time_opencv = timingExec(start, end);
    
    cv::Mat temp;
    dst_opencvD.download(dst_opencv);

    cout<< "dst_opencv step  " <<  dst_opencv.step << endl;
    //warm up
    HPCStatus_t sta = gaussianBlur<Tsrc,filterSize,ncSrc, ncDst, nc>(src, &dst);
    gettimeofday(&start, NULL);
    for(int i =0;i<N;i++){
        sta = gaussianBlur<Tsrc,filterSize,ncSrc, ncDst, nc>(src, &dst);
        cudaDeviceSynchronize();
    }
    gettimeofday(&end, NULL);
    EXPECT_EQ(HPC_SUCCESS, sta);

    double time_fastcv = timingExec(start, end);


        //warm up
    HPCStatus_t sta_x86 = gaussianBlur<Tsrc,filterSize,ncSrc, ncDst, nc>(src_x86, &dst_x86);
    gettimeofday(&start, NULL);
    sta_x86 = gaussianBlur<Tsrc,filterSize,ncSrc, ncDst, nc>(src_x86, &dst_x86);
    gettimeofday(&end, NULL);
    EXPECT_EQ(HPC_SUCCESS, sta_x86);

    double time_fastcv_x86 = timingExec(start, end);

    printf("fastcv_cuda cost %f fastcv_x86 cost %f   opencv cost %f\n", time_fastcv,time_fastcv_x86, time_opencv);
#else
   struct timeval start, end;
    // call opencv to verify result
    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, ncSrc), input_data, step);
    cv::Mat dst_opencv(output_height, output_width,
            CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst), output_data_opencv, dstep);
    cv::Size ksize;
    ksize.height = filterSize;
    ksize.width  = filterSize;

    //warm up
    cv::GaussianBlur(src_opencv,dst_opencv,ksize,-1,-1);
    gettimeofday(&start, NULL);
    cv::GaussianBlur(src_opencv,dst_opencv,ksize,-1,-1);
    gettimeofday(&end, NULL);
    double time_opencv = timingExec(start, end);

    //warm up
    HPCStatus_t sta = gaussianBlur<Tsrc,filterSize,ncSrc, ncDst, nc>(src, &dst);
    gettimeofday(&start, NULL);
    sta = gaussianBlur<Tsrc,filterSize,ncSrc, ncDst, nc>(src, &dst);
    gettimeofday(&end, NULL);
    EXPECT_EQ(HPC_SUCCESS, sta);

    double time_fastcv = timingExec(start, end);

    printf("fastcv cost %f  opencv cost %f\n", time_fastcv, time_opencv);

#endif

      // check results
#if defined(FASTCV_USE_CUDA) || defined(FASTCV_USE_OCL)
       // size_t output_data_num = output_width*output_height*output_channels;
        //Tdst* dst_data_x86 = (Tdst*)malloc(output_data_num*sizeof(Tdst));
        memset(output_data,0,output_height*dstep);
        dst.toHost(output_data);
        checkResult<Tdst, ncDst>(output_data, dst_opencv.ptr<Tdst>(), output_height, output_width, dstep/sizeof(Tdst), diff_THR);

       // checkResult<Tdst, ncDst>((Tdst*)dst_x86.ptr(), dst_opencv.ptr<Tdst>(), output_height, output_width, dstep/sizeof(Tdst), diff_THR);
        checkResult<Tdst, ncDst>(output_data, dst_opencv.ptr<Tdst>(), output_height, output_width, dstep/sizeof(Tdst), diff_THR);

       // free(dst_data_x86);
#elif defined(FASTCV_USE_OCL)
        memset(output_data,0,output_height*dstep);
        dst.toHost(output_data);
        checkResult<Tdst, ncDst>(output_data, dst_opencv.ptr<Tdst>(), output_height, output_width, dstep/sizeof(Tdst), diff_THR);
#else
        checkResult<Tdst, ncDst>((Tdst*)dst.ptr(), dst_opencv.ptr<Tdst>(), output_height, output_width, dstep/sizeof(Tdst), diff_THR);
#endif

    free(input_data);
    free(output_data);
    free(output_data_opencv);
    return 0;
}

TEST_F(gaussianBlurTest, uchar){
      imageParam p ;
   //   ASSERT_TRUE(!imageparams.empty());
    //  for(int i =0; i<imageparams.size(); i++)
    //  {
          p = imagep;
          testGaussianBlurFunc<uchar, 1, uchar , 1, 1, 3, 1>(p,1.01);
          testGaussianBlurFunc<uchar, 3, uchar , 3, 3, 3, 1>(p,1.01);
          testGaussianBlurFunc<uchar, 4, uchar , 4, 4, 3, 1>(p,1.01);
          testGaussianBlurFunc<uchar, 1, uchar , 1, 1, 5, 1>(p,1.01);
          testGaussianBlurFunc<uchar, 3, uchar , 3, 3, 5, 1>(p,1.01);
          testGaussianBlurFunc<uchar, 4, uchar , 4, 4, 5, 1>(p,1.01);
          testGaussianBlurFunc<uchar, 1, uchar , 1, 1, 7, 1>(p,1.01);
          testGaussianBlurFunc<uchar, 3, uchar , 3, 3, 7, 1>(p,1.01);
          testGaussianBlurFunc<uchar, 4, uchar , 4, 4, 7, 1>(p,1.01);
    //  }
     

}

TEST_F(gaussianBlurTest, float){
    imageParam p ;
  //  ASSERT_TRUE(!imageparams.empty());
  //  for(int i =0; i<imageparams.size(); i++)
  //  {
        p = imagep;
        testGaussianBlurFunc<float, 1, float , 1, 1, 3, 255>(p,1e-5);
        testGaussianBlurFunc<float, 3, float , 3, 3, 3, 255>(p,1e-5);
        testGaussianBlurFunc<float, 4, float , 4, 4, 3, 255>(p,1e-5);
        testGaussianBlurFunc<float, 1, float , 1, 1, 5, 255>(p,1e-5);
        testGaussianBlurFunc<float, 3, float , 3, 3, 5, 255>(p,1e-5);
        testGaussianBlurFunc<float, 4, float , 4, 4, 5, 255>(p,1e-5);
        testGaussianBlurFunc<float, 1, float , 1, 1, 7, 255>(p,1e-5);
        testGaussianBlurFunc<float, 3, float , 3, 3, 7, 255>(p,1e-5);
        testGaussianBlurFunc<float, 4, float , 4, 4, 7, 255>(p,1e-5);
  //  }
}


