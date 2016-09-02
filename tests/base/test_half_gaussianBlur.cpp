#include "test_head.hpp"
#include <opencv2/gpu/gpu.hpp>
#include<opencv2/ocl/ocl.hpp>


#define N 1000
//#define PRINT_ALL

using namespace std;
using namespace HPC::fastcv;
using namespace uni::half;

struct imageParam{
    size_t srcWidth;
    size_t srcHeight;
    size_t dstWidth;
    size_t dstHeight;
    int spadding;
    int dpadding;
};



class halfgaussianBlurTest : public testing::Test
{
    public:
        virtual void SetUp()
        {
            std::vector<imageParam>::iterator iter = imageparams.begin();
            if(globalEnvironment::Instance()->testconfig.hasSection("halfgaussianBlurTest"))
            {
                Cfg::cfg_type vec_arg = globalEnvironment::Instance()->testconfig.sectionConfigVec("halfgaussianBlurTest");
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
}



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
void checkResult(const T *data1, const float *data2, const int height, const int width, int dstep, const float diff_THR) {
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

/*
int main(int argc, char *argv[]){
    if(argc != 1 && argc != 7){
        printf("Error params!\n");
        printf("Usage: ./run src_width src_height \
                dst_width dst_height \n");
    }
     imagep={12, 12, 12,12,7,11};
   //  imagep={1024, 1024, 1024, 1024};
  //     imagep={450, 660, 450, 660};
   //  imagep={1280, 720, 1280, 720};
     imagep={1920, 1080, 1920, 1080, 0, 0};
    //imagep={4096, 4096, 4096, 4096};
//  imagep={4200, 3136, 4200, 3136};
 // imagep={5000, 5000, 5000, 5000};
 // imagep={10000, 10000, 10000, 10000};

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
template<typename Tsrc,int ncSrc, typename Tdst ,int ncDst, int nc, int filterSize, int val_div>
int testGaussianBlurFunc(imageParam p,float diff_THR) {
    size_t input_width = p.srcWidth;
    size_t input_height = p.srcHeight;
    size_t input_channels = ncSrc;
    size_t output_width = p.dstWidth;
    size_t output_height = p.dstHeight;
    size_t output_channels = ncDst;
    int spadding = imagep.spadding;
    int dpadding = imagep.dpadding;

    printf("Input Mat scale: width*height*channels = %ld*%ld*%ld\n", input_width, input_height, input_channels);
    printf("Output Mat scale: width*height*channels = %ld*%ld*%ld\n", output_width, output_height, output_channels);

    int inStride =  (input_width * input_channels + spadding);
    size_t step = inStride * sizeof(Tsrc);
    size_t step_opencv = inStride * sizeof(float);
    Tsrc* input_data = (Tsrc*)malloc(input_height*step);
    float* input_data_opencv = (float*)malloc(input_height*step_opencv);
    randomPaddingData<Tsrc, val_div>(input_data, input_height,
            input_width*input_channels, input_width * input_channels + spadding);

    for(int i=0;i<input_height * inStride;i++){
        input_data_opencv[i] = input_data[i];
    }

#if defined(USE_CUDA) || defined(USE_OCL)
        Mat<Tsrc, ncSrc> src(input_height, input_width, step);
        src.fromHost(input_data);
#else
        const Mat<Tsrc, ncSrc> src(input_height, input_width, input_data, step);
#endif

        int outStride = (output_width * output_channels + dpadding);
        size_t dstep = outStride * sizeof(Tdst);
        size_t dstep_opencv = outStride * sizeof(float);
        Tdst* output_data = (Tdst*)malloc(output_height*dstep);
        float* output_data_opencv = (float*)malloc(output_height*dstep_opencv);

#if defined(USE_CUDA)
        Mat<Tdst, ncDst> dst(output_height, output_width, dstep);
        dst.fromHost(output_data);
#elif defined(USE_OCL)
        Mat<Tdst, ncDst> dst(output_height, output_width, dstep);
        dst.fromHost(output_data);
#else
        Mat<Tdst, ncDst> dst(output_height, output_width, output_data, dstep);
#endif


#if defined(USE_OCL)
    struct timeval start, end;
    // call opencv to verify result
    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<float>::depth, ncSrc), input_data_opencv, step);
    cv::Mat dst_opencv(output_height, output_width,
            CV_MAKETYPE(cv::DataType<float>::depth, ncDst), output_data_opencv, dstep);
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

#elif defined(USE_CUDA)
    struct timeval start, end;
    // call opencv to verify result
    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<float>::depth, ncSrc), input_data_opencv, step_opencv);
    cv::Mat dst_opencv(output_height, output_width,
            CV_MAKETYPE(cv::DataType<float>::depth, ncDst), output_data_opencv, dstep_opencv);
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

    printf("fastcv cost %f  opencv cost %f\n", time_fastcv, time_opencv);
#else

#endif

      // check results
#if defined(USE_CUDA) || defined(USE_OCL)
        memset(output_data,0,output_height*dstep);
        dst.toHost(output_data);
        checkResult<Tdst, ncDst>(output_data, dst_opencv.ptr<float>(), output_height, output_width, dstep/sizeof(Tdst), diff_THR);
#else
        checkResult<Tdst, ncDst>(dst.ptr(), dst_opencv.ptr<Tdst>(), output_height, output_width, dstep/sizeof(Tdst), diff_THR);
#endif

    free(input_data);
    free(input_data_opencv);
    free(output_data);
    free(output_data_opencv);
    return 0;
}

TEST_F(halfgaussianBlurTest, float){
     imageParam p ;
      ASSERT_TRUE(!imageparams.empty());
      for(int i =0; i<imageparams.size(); i++)
      {
          p = imageparams[i];
          testGaussianBlurFunc<half_t, 1, half_t , 1, 1, 3, 255>(p,1e-5);
        //    testGaussianBlurFunc<float, 1, float , 1, 1, 3, 255>(p,1e-5);
          testGaussianBlurFunc<half_t, 3, half_t , 3, 3, 3, 255>(p,1e-5);
        //    testGaussianBlurFunc<float, 3, float , 3, 3, 3, 255>(p,1e-5);
            testGaussianBlurFunc<half_t, 4, half_t , 4, 4, 3, 255>(p,1e-5);
        //    testGaussianBlurFunc<float, 4, float , 4, 4, 3, 255>(p,1e-5);
            testGaussianBlurFunc<half_t, 1, half_t , 1, 1, 5, 255>(p,1e-5);
            testGaussianBlurFunc<half_t, 3, half_t , 3, 3, 5, 255>(p,1e-5);
            testGaussianBlurFunc<half_t, 4, half_t , 4, 4, 5, 255>(p,1e-5);
            testGaussianBlurFunc<half_t, 1, half_t , 1, 1, 7, 255>(p,1e-5);
            testGaussianBlurFunc<half_t, 3, half_t , 3, 3, 7, 255>(p,1e-5);
            testGaussianBlurFunc<half_t, 4, half_t , 4, 4, 7, 255>(p,1e-5);
         
      }
   
}


