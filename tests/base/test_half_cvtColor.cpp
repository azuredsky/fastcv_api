#include "test_head.hpp"
#include <opencv2/gpu/gpu.hpp>
#include<opencv2/ocl/ocl.hpp>
using namespace HPC::fastcv;
using namespace uni::half;
#define N 1000

//#define PRINT_ALL
//#define begin           gettimeofday(&start, NULL);for(int i=0;i<N;i++){
//#define stop            }cv::ocl::finish();gettimeofday(&end, NULL);time_opencv = timingExec(start, end); printf("\t opencv cost = %f ms\t\n ", time_opencv);
struct imageParam{
    size_t srcWidth;
    size_t srcHeight;
    size_t dstWidth;
    size_t dstHeight;
    int spadding;
    int dpadding;
};

/*
ColorCvtType type1 = BGR2NV21;

ColorCvtType type2 = NV212BGR;
ColorCvtType type3 = YCrCb2BGR;
ColorCvtType type4 = BGR2YCrCb;
ColorCvtType type5 = HSV2BGR;
ColorCvtType type6 = BGR2HSV;
ColorCvtType type7 = BGR2GRAY;
ColorCvtType type8 = GRAY2BGR;
ColorCvtType type9 = BGRA2BGR;
ColorCvtType type10 = BGR2BGRA;
ColorCvtType type11 = BGR2NV12;
ColorCvtType type12 = NV122BGR;
ColorCvtType type13 = NV212RGB;
ColorCvtType type14 = RGB2NV21;
ColorCvtType type15 = BGR2RGB;
ColorCvtType type16 = BGR2LAB;
ColorCvtType type17 = LAB2BGR;
ColorCvtType type18 = NV122RGB;
ColorCvtType type19 = RGB2NV12;
*/
class halfcvtColorTest : public testing::Test
{
     public:
        virtual void SetUp()
        {
            std::vector<imageParam>::iterator iter = imageparams.begin();
            if(globalEnvironment::Instance()->testconfig.hasSection("halfcvtColorTest"))
            {
                Cfg::cfg_type vec_arg = globalEnvironment::Instance()->testconfig.sectionConfigVec("halfcvtColorTest");
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
                imagep={64, 64, 64, 64};
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
void checkResult_colortype(const T *data1, const T *data2, const int height, const int width, const float diff_THR, ColorCvtType type) {
    double max = DBL_MIN;
    double min = DBL_MAX;
    double ave_diff = 0.0;
    double l1_diff  = 0.0;
    double l2_diff  = 0.0;
    double temp_diff = 0.0;
    double ave_data = 0.0;
    double order = 0.0;

    int num = height*width *nc;
    for(int i=0; i < num; i++){
        //ASSERT_NEAR(data1[i], data2[i], 1e-5);
        temp_diff = fabs((double) *(data1 + i)-(double) *(data2 + i));

        if(type == BGR2HSV) {
            if(temp_diff > 100) temp_diff = fabs(temp_diff - 180);
        }

#ifdef PRINT_ALL

if(temp_diff>diff_THR)
 std::cout << "Print data12: " << i << " : " << (float) data1[i] << " : " << (float) data2[i] << " : " << temp_diff << std::endl;
 //std::cout << "Print data12: " << i << " : " << (float) data1[i] << " : " << (float) data2[i] << " : " << temp_diff << std::endl;
#endif
        max = (temp_diff > max) ? temp_diff : max;
        min = (temp_diff < min) ? temp_diff : min;
        ave_data += fabs((double)data1[i]);
        l1_diff += temp_diff;
        l2_diff += temp_diff*temp_diff;
    }
    ave_data = ave_data / num;
    ave_diff = l1_diff / num;
    order = (ave_data == 0) ? 0.0 : ave_diff / ave_data;
    l2_diff = sqrt(l2_diff);
/*
#ifdef PRINT_ALL
    std::cout << "Min Diff = " << min << " , Max Diff = " << max
    << " , Ave Diff = " << ave_diff << " , Ave_Diff / Ave_Data = " << order
    << " , [L1, L2] = [ " << l1_diff << ", " << l2_diff << " ]" << std::endl;
#endif
 */
    EXPECT_LT(max, diff_THR);
    //ASSERT_NEAR(order, 1e-5, 1e-5);
}

template<typename T, int nc>
void checkResult_colortype(const T *data1, const float *data2, const int height,
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

            if(type == BGR2HSV) {
                if(temp_diff > 100) temp_diff = fabs(temp_diff - 180);
            }

#ifdef PRINT_ALL

            if(temp_diff>diff_THR)
                std::cout << "Print data12: " << i << " : " << j << " : " << (float) data1[i] << " : " << (float) data2[i] << " : " << temp_diff << std::endl;
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
    /*
#ifdef PRINT_ALL
std::cout << "Min Diff = " << min << " , Max Diff = " << max
<< " , Ave Diff = " << ave_diff << " , Ave_Diff / Ave_Data = " << order
<< " , [L1, L2] = [ " << l1_diff << ", " << l2_diff << " ]" << std::endl;
#endif
     */
    EXPECT_LT(max, diff_THR);
    //ASSERT_NEAR(order, 1e-5, 1e-5);
}

template<typename T, int nc>
void checkResult_NV21(const T *data1, const T *data2, const int height, const int width, int sstep, const float diff_THR) {
    double max = DBL_MIN;
    double min = DBL_MAX;
    double temp_diff0 = 0.0;
    double temp_diff1 = 0.0;
    double temp_diff2 = 0.0;

    int num = height*width *nc;


    for(int j = 0; j < height; j += 2 ) {
        for(int i = 0; i < width*nc; i += nc){
            if ( i % (nc*2) == 0) {
                temp_diff0 = fabs((double) *(data1 + i)-(double) *(data2 + i));
                max = (temp_diff0 > max) ? temp_diff0 : max;
                min = (temp_diff0 < min) ? temp_diff0 : min;
                temp_diff1 = fabs((double) *(data1 + i + 1)-(double) *(data2 + i + 1));
                max = (temp_diff1 > max) ? temp_diff1 : max;
                min = (temp_diff1 < min) ? temp_diff1 : min;
                temp_diff2 = fabs((double) *(data1 + i + 2)-(double) *(data2 + i + 2));
                max = (temp_diff2 > max) ? temp_diff2 : max;
                min = (temp_diff2 < min) ? temp_diff2 : min;

#ifdef PRINT_ALL
       //         if(temp_diff0 > diff_THR || temp_diff1 > diff_THR || temp_diff2 > diff_THR){
                    std::cout << "Print B: " << i << " : " << (float) data1[i] << " : " << (float) *(data2 + i) << " : " << temp_diff0 << " : end" << std::endl;
                    std::cout << "Print G: " << i+1 << " : " << (float) data1[i+1] << " : " << (float) *(data2 + i + 1)  << " : " << temp_diff1 << " : end" << std::endl;
                    std::cout << "Print R: " << i+2 << " : " << (float) data1[i+2] << " : " << (float) *(data2 + i + 2)  << " : " << temp_diff2 << " : end" << std::endl;
            //    }
#endif
            }
        }
        data1 += 2*sstep;
        data2 += 2*sstep;
    }

    EXPECT_LT(max, diff_THR);
    //ASSERT_NEAR(order, 1e-5, 1e-5);
}

/*
int main(int argc, char *argv[]){
    if(argc != 1 && argc != 7){
        printf("Error params!\n");
        printf("Usage: ./run src_width src_height dst_width dst_height \n");
    }
     imagep={64, 64, 64, 64};
     imagep={640, 1040, 640, 1040};
    //   imagep={1280, 720, 1280, 720};
    //  imagep={20, 10, 20, 10, 6, 8};
  //   imagep={1920, 1080, 1920, 1080, 7, 11};
   // imagep={4200, 3136, 4200, 3136};
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


    int inStride = (input_width * input_channels + spadding);
    size_t step = inStride*sizeof(Tsrc);
    Tsrc* input_data = (Tsrc*)malloc(input_height*step);
    Tsrc* input_data1 = (Tsrc*)malloc(input_height*step);

    randomPaddingData<Tsrc, val_div>(input_data, input_height,
            input_width*input_channels, input_width * input_channels + spadding);


    size_t step_opencv = inStride * sizeof(float) ;
    float* input_data_opencv = (float*)malloc(input_height*step_opencv);

    for(int i=0;i<input_height * inStride;i++){
        input_data_opencv[i] = input_data[i];
    }



    struct timeval start, end;

    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<float>::depth, ncSrc), input_data_opencv, step_opencv);
    if( type == HSV2BGR) {
        cv::cvtColor(src_opencv, src_opencv, CV_BGR2HSV);
    }

    if( type == LAB2BGR) {
        cv::cvtColor(src_opencv, src_opencv, CV_BGR2Lab);
    }
    size_t height_tmp0 = (type == BGR2NV21 || type == BGR2NV12 || type == RGB2NV21 || type == RGB2NV12) ? output_height*3/2 : output_height;

    int outStride =  (output_width * output_channels + dpadding);
    size_t dstep = outStride*sizeof(Tdst);
    Tdst* output_data = (Tdst*)malloc(height_tmp0 * dstep);
    size_t dstep_opencv = outStride * sizeof(float);
    float* output_data_opencv = (float*)malloc(height_tmp0* dstep_opencv);

#if defined(USE_CUDA)
    Mat<Tdst, ncDst> dst(height_tmp0, output_width, dstep);
    dst.fromHost(output_data);
#elif defined(USE_OCL)
    Mat<Tdst, ncDst> dst(height_tmp0, output_width, dstep);
    dst.fromHost(output_data);
#else
    Mat<Tdst, ncDst> dst(height_tmp0, output_width, output_data, dstep);
#endif



#if defined(USE_CUDA)

    for(int i=0;i<input_height * inStride;i++){
        input_data[i] = input_data_opencv[i];
    }
    Mat<Tsrc, ncSrc> src(input_height, input_width, step);
    src.fromHost(input_data);
    //warm up
    HPCStatus_t sta = cvtColor<type, Tsrc, ncSrc, Tdst, ncDst>(src, &dst);

    gettimeofday(&start, NULL);
    for(int i=0;i<N;i++){
        sta = cvtColor<type, Tsrc, ncSrc, Tdst, ncDst>(src, &dst);
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    double time_fastcv = timingExec(start, end);
    double time_opencv;
    EXPECT_EQ(HPC_SUCCESS, sta);

    Mat<Tsrc, ncSrc> dst_fastcv(input_height, input_width, step);

    cv::gpu::GpuMat src_opencvD(src_opencv);

    cv::gpu::GpuMat dst_opencvD;
    cv::gpu::GpuMat dst_opencvD2;
    cv::Mat dst_opencv;


    //cv::Mat dst_opencv;//(input_height,input_width);
    switch(type) {
        case YCrCb2BGR:
            printf("YCrCb2BGR :  fastcv cost = %f ms\t\t ", time_fastcv);
            //warm up
            cv::gpu::cvtColor(src_opencvD, dst_opencvD, CV_YCrCb2BGR);

            gettimeofday(&start, NULL);
            for(int i=0;i<N;i++){
                cv::gpu::cvtColor(src_opencvD, dst_opencvD2, CV_YCrCb2BGR);
                if(4==ncDst)cv::gpu::cvtColor(dst_opencvD2, dst_opencvD, CV_BGR2BGRA);
            }
            gettimeofday(&end, NULL);
            time_opencv = timingExec(start, end);

            dst_opencvD.download(dst_opencv);

            break;
        case BGR2YCrCb:
            printf("BGR2YCrCb :  fastcv cost = %f ms\t\t ", time_fastcv);
            //warm up
            cv::gpu::cvtColor(src_opencvD, dst_opencvD, CV_BGR2YCrCb);

            gettimeofday(&start, NULL);
            for(int i=0;i<N;i++)
                cv::gpu::cvtColor(src_opencvD, dst_opencvD, CV_BGR2YCrCb);
            gettimeofday(&end, NULL);
            time_opencv = timingExec(start, end);

            dst_opencvD.download(dst_opencv);
            break;
        case LAB2BGR:
            printf("LAB2BGR :  fastcv cost = %f ms\t\t ", time_fastcv);
            //warm up
            cv::gpu::cvtColor(src_opencvD, dst_opencvD, CV_Lab2BGR);

            gettimeofday(&start, NULL);
            for(int i=0;i<N;i++){
                cv::gpu::cvtColor(src_opencvD, dst_opencvD2, CV_Lab2BGR);
                cudaDeviceSynchronize();
                if(ncDst == 4) cv::gpu::cvtColor(dst_opencvD2, dst_opencvD, cv::COLOR_BGR2BGRA);
                cudaDeviceSynchronize();
            }
            gettimeofday(&end, NULL);
            time_opencv = timingExec(start, end);
            dst_opencvD.download(dst_opencv);
            break;
        case BGR2LAB:
            printf("BGR2LAB :  fastcv cost = %f ms\t\t ", time_fastcv);
            //warm up
            if(ncSrc == 4) cv::cvtColor(src_opencv, src_opencv, cv::COLOR_BGRA2BGR);
            cv::cvtColor(src_opencv, dst_opencv, CV_BGR2Lab);

            gettimeofday(&start, NULL);
            for(int i=0;i<N;i++){
                if(ncSrc == 4) cv::cvtColor(src_opencv, src_opencv, cv::COLOR_BGRA2BGR);
                cv::gpu::cvtColor(src_opencvD, dst_opencvD, CV_BGR2Lab);
                cudaDeviceSynchronize();
            }
            gettimeofday(&end, NULL);
            time_opencv = timingExec(start, end);
            dst_opencvD.download(dst_opencv);
            break;
        case BGR2GRAY:
            printf("BGR2GRAY :  fastcv cost = %f ms\t\t ", time_fastcv);
            gettimeofday(&start, NULL);
            for(int i=0;i<N;i++){
                if(ncSrc == 3) cv::gpu::cvtColor(src_opencvD, dst_opencvD, CV_BGR2GRAY);
                else if(ncSrc == 4) cv::gpu::cvtColor(src_opencvD, dst_opencvD, CV_BGRA2GRAY);
            }
            gettimeofday(&end, NULL);
            time_opencv = timingExec(start, end);
            dst_opencvD.download(dst_opencv);
            break;
        case GRAY2BGR:
            printf("GRAY2BGR :  fastcv cost = %f ms\t\t ", time_fastcv);
            gettimeofday(&start, NULL);
            for(int i=0;i<N;i++){
                if(ncDst == 3) cv::gpu::cvtColor(src_opencvD, dst_opencvD, CV_GRAY2BGR);
                else if(ncDst == 4) cv::gpu::cvtColor(src_opencvD, dst_opencvD, CV_GRAY2BGRA);
            }
            gettimeofday(&end, NULL);
            time_opencv = timingExec(start, end);
            dst_opencvD.download(dst_opencv);
            break;
        case BGR2RGB:
            printf("BGR2RGB :  fastcv cost = %f ms\t\t ", time_fastcv);
            gettimeofday(&start, NULL);
            for(int i=0;i<N;i++){
                if(ncSrc == 3 && ncDst == 3) cv::gpu::cvtColor(src_opencvD, dst_opencvD, CV_BGR2RGB);
                else if(ncSrc == 3 && ncDst == 4) cv::gpu::cvtColor(src_opencvD, dst_opencvD, CV_BGR2RGBA);
                else if(ncSrc == 4 && ncDst == 3) cv::gpu::cvtColor(src_opencvD, dst_opencvD, CV_BGRA2RGB);
                else if(ncSrc == 4 && ncDst == 4) cv::gpu::cvtColor(src_opencvD, dst_opencvD, CV_BGRA2RGBA);
                cudaDeviceSynchronize();
            }
            gettimeofday(&end, NULL);
            time_opencv = timingExec(start, end);
            dst_opencvD.download(dst_opencv);
            break;
        case BGR2BGRA:
            printf("BGR2BGRA :  fastcv cost = %f ms\t\t ", time_fastcv);

            //warm up
            cv::gpu::cvtColor(src_opencvD, dst_opencvD, cv::COLOR_BGR2BGRA);
            gettimeofday(&start, NULL);
            for(int i=0;i<N;i++){
                cv::gpu::cvtColor(src_opencvD, dst_opencvD, cv::COLOR_BGR2BGRA);
                cudaDeviceSynchronize();
            }
            gettimeofday(&end, NULL);
            time_opencv = timingExec(start, end);

            dst_opencvD.download(dst_opencv);
            break;
        case BGRA2BGR:
            printf("BGRA2BGR :  fastcv cost = %f ms\t\t ", time_fastcv);
            //warm up
            cv::gpu::cvtColor(src_opencvD, dst_opencvD, cv::COLOR_BGRA2BGR);
            gettimeofday(&start, NULL);
            for(int i=0;i<N;i++){
                cv::gpu::cvtColor(src_opencvD, dst_opencvD, cv::COLOR_BGRA2BGR);
                cudaDeviceSynchronize();
            }
            gettimeofday(&end, NULL);
            time_opencv = timingExec(start, end);
            dst_opencvD.download(dst_opencv);
            break;
        case HSV2BGR:
            printf("HSV2BGR :  fastcv cost = %f ms..\t\t ", time_fastcv);
            cv::gpu::cvtColor(src_opencvD, dst_opencvD, CV_HSV2BGR);
            gettimeofday(&start, NULL);
            for(int i=0;i<N;i++){
                cv::gpu::cvtColor(src_opencvD, dst_opencvD2, CV_HSV2BGR);
                if(ncDst == 4) {
                    cv::gpu::cvtColor(dst_opencvD2, dst_opencvD, CV_BGR2BGRA);
                }
            }
            gettimeofday(&end, NULL);
            time_opencv = timingExec(start, end);

            dst_opencvD.download(dst_opencv);
            break;
        case BGR2HSV:
            printf("BGR2HSV :  fastcv cost = %f ms\t\t ", time_fastcv);
        //  cv::gpu::cvtColor(src_opencvD, dst_opencvD, CV_BGR2HSV);
            cv::gpu::cvtColor(src_opencvD, dst_opencvD, CV_BGR2HSV);
            gettimeofday(&start, NULL);
            for(int i=0;i<N;i++){
            //    if(ncSrc == 4) cv::gpu::cvtColor(src_opencvD, dst_opencvD, CV_BGR2HSV);
                cv::gpu::cvtColor(src_opencvD, dst_opencvD, CV_BGR2HSV);
            }
            gettimeofday(&end, NULL);
            time_opencv = timingExec(start, end);

            dst_opencvD.download(dst_opencv);
            break;
        case BGR2NV21:
            sta = cvtColor<NV212BGR, Tdst, ncDst, Tsrc, ncSrc>(dst, &dst_fastcv);
            EXPECT_EQ(HPC_SUCCESS, sta);
            break;
        case BGR2NV12:
            gettimeofday(&start, NULL);
            sta = cvtColor<NV122BGR, Tdst, ncDst, Tsrc, ncSrc>(dst, &dst_fastcv);
            cudaDeviceSynchronize();
            gettimeofday(&end, NULL);
            time_fastcv = timingExec(start, end);
            EXPECT_EQ(HPC_SUCCESS, sta);
            printf("NV212BGR:  fastcv cost = %f ms\n", time_fastcv);
            break;
        case RGB2NV21:
            sta = cvtColor<NV212RGB, Tdst, ncDst, Tsrc, ncSrc>(dst, &dst_fastcv);
            EXPECT_EQ(HPC_SUCCESS, sta);
            break;
        case RGB2NV12:
            sta = cvtColor<NV122RGB, Tdst, ncDst, Tsrc, ncSrc>(dst, &dst_fastcv);
            EXPECT_EQ(HPC_SUCCESS, sta);
            break;
        default:
            return HPC_NOT_SUPPORTED;
    }
    printf("\t opencv cost = %f ms\t\n ", time_opencv);

#elif defined(USE_OCL)
    Mat<Tsrc, ncSrc> src(input_height, input_width, step);
    src.fromHost(input_data);
    HPCStatus_t sta ;
    sta = cvtColor<type, Tsrc, ncSrc, Tdst, ncDst>(src, &dst);
    gettimeofday(&start, NULL);
    for(int k=0;k < N; k++){
        sta = cvtColor<type, Tsrc, ncSrc, Tdst, ncDst>(src, &dst);
        clFinish(opencl.getCommandQueue());
    }

    gettimeofday(&end, NULL);
    double time_fastcv = timingExec(start, end);
    double time_opencv;
    EXPECT_EQ(HPC_SUCCESS, sta);
    Mat<Tsrc, ncSrc> dst_fastcv(input_height, input_width, step);

    cv::ocl::oclMat src_opencvD(src_opencv);
    cv::ocl::oclMat dst_opencvD;
    cv::Mat dst_opencv;
    switch(type) {
        case YCrCb2BGR:
            printf("YCrCb2BGR :  fastcv cost = %f ms\t\t ", time_fastcv);
            //warm up
            cv::ocl::cvtColor(src_opencvD, dst_opencvD, CV_YCrCb2BGR, ncDst);
            begin
            cv::ocl::cvtColor(src_opencvD, dst_opencvD, CV_YCrCb2BGR, ncDst);
            stop
            break;
        case BGR2YCrCb:
            printf("BGR2YCrCb :  fastcv cost = %f ms\t\t ", time_fastcv);
            //warm up
            cv::ocl::cvtColor(src_opencvD, dst_opencvD, CV_BGR2YCrCb);

            begin
            cv::ocl::cvtColor(src_opencvD, dst_opencvD, CV_BGR2YCrCb);
            stop
            break;
        case LAB2BGR:
            printf("LAB2BGR :  fastcv cost = %f ms\t\t ", time_fastcv);
            //warm up
            cv::ocl::cvtColor(src_opencvD, dst_opencvD, CV_Lab2BGR,ncDst);
            begin
            cv::ocl::cvtColor(src_opencvD, dst_opencvD, CV_Lab2BGR,ncDst);
            stop
            break;
        case BGR2LAB:
            if(ncSrc == 4) cv::cvtColor(src_opencvD, src_opencvD, cv::COLOR_BGRA2BGR);
            cv::cvtColor(src_opencvD, dst_opencvD, CV_BGR2Lab);
            break;
        case BGR2GRAY:
            printf("BGR2GRAY :  fastcv cost = %f ms\t\t ", time_fastcv);
            //warm up
            if(ncSrc == 3) cv::ocl::cvtColor(src_opencvD, dst_opencvD, CV_BGR2GRAY);
            else if(ncSrc == 4) cv::ocl::cvtColor(src_opencvD, dst_opencvD, CV_BGRA2GRAY);

            begin
            if(ncSrc == 3) cv::ocl::cvtColor(src_opencvD, dst_opencvD, CV_BGR2GRAY);
            else if(ncSrc == 4) cv::ocl::cvtColor(src_opencvD, dst_opencvD, CV_BGRA2GRAY);
            stop
            break;
        case GRAY2BGR:
            printf("GRAY2BGR :  fastcv cost = %f ms\t\t ", time_fastcv);
            //warm up
            if(ncDst == 3) cv::ocl::cvtColor(src_opencvD, dst_opencvD, CV_GRAY2BGR);
            else if(ncDst == 4) cv::ocl::cvtColor(src_opencvD, dst_opencvD, CV_GRAY2BGRA);

            begin
            if(ncDst == 3) cv::ocl::cvtColor(src_opencvD, dst_opencvD, CV_GRAY2BGR);
            else if(ncDst == 4) cv::ocl::cvtColor(src_opencvD, dst_opencvD, CV_GRAY2BGRA);
            stop

            break;
        case BGR2RGB:
            printf("BGR2RGB :  fastcv cost = %f ms\t\t ", time_fastcv);
            //warm up
            if(ncSrc == 3 && ncDst == 3) cv::cvtColor(src_opencv, dst_opencv, CV_BGR2RGB);
            else if(ncSrc == 3 && ncDst == 4) cv::cvtColor(src_opencv, dst_opencv, CV_BGR2RGBA);
            else if(ncSrc == 4 && ncDst == 3) cv::cvtColor(src_opencv, dst_opencv, CV_BGRA2RGB);
            else if(ncSrc == 4 && ncDst == 4) cv::cvtColor(src_opencv, dst_opencv, CV_BGRA2RGBA);
            break;
        case BGR2BGRA:
            printf("BGR2BGRA :  fastcv cost = %f ms\t\t ", time_fastcv);
            //warm up
            cv::ocl::cvtColor(src_opencvD, dst_opencvD, cv::COLOR_BGR2BGRA);

            begin
            cv::ocl::cvtColor(src_opencvD, dst_opencvD, cv::COLOR_BGR2BGRA);
            stop

            break;
        case BGRA2BGR:
            printf("BGRA2BGR :  fastcv cost = %f ms\t\t ", time_fastcv);
            //warm up
            cv::ocl::cvtColor(src_opencvD, dst_opencvD, cv::COLOR_BGRA2BGR);
            begin
            cv::ocl::cvtColor(src_opencvD, dst_opencvD, cv::COLOR_BGRA2BGR);
            stop
            break;
        case HSV2BGR:
            printf("HSV2BGR :  fastcv cost = %f ms\t\t ", time_fastcv);
            //warm up
            cv::ocl::cvtColor(src_opencvD, dst_opencvD, CV_HSV2BGR,ncDst);
            begin
            cv::ocl::cvtColor(src_opencvD, dst_opencvD, CV_HSV2BGR,ncDst);
            stop
            break;
        case BGR2HSV:
            printf("BGR2HSV :  fastcv cost = %f ms\t\t ", time_fastcv);
            cv::ocl::cvtColor(src_opencvD, dst_opencvD, CV_BGR2HSV);
            gettimeofday(&start, NULL);

            for(int i=0;i<N;i++){
                cv::ocl::cvtColor(src_opencvD, dst_opencvD, CV_BGR2HSV);
                cv::ocl::finish();
            }

            gettimeofday(&end, NULL);
            time_opencv = timingExec(start, end);
            printf("\t opencv cost = %f ms\t\n ", time_opencv);

            break;
        case BGR2NV21:
            sta = cvtColor<NV212BGR, Tdst, ncDst, Tsrc, ncSrc>(dst, &dst_fastcv);
            EXPECT_EQ(HPC_SUCCESS, sta);
            break;
        case BGR2NV12:
            gettimeofday(&start, NULL);
          //  sta = cvtColor<Tdst, ncDst, Tsrc, ncSrc, NV122BGR>(dst, &dst_fastcv);
            sta = cvtColor<NV122BGR, Tdst, ncDst, Tsrc, ncSrc>(dst, &dst_fastcv);

            gettimeofday(&end, NULL);
            time_fastcv = timingExec(start, end);
            EXPECT_EQ(HPC_SUCCESS, sta);
            printf("NV212BGR:  fastcv cost = %f ms\n", time_fastcv);
            break;
        case RGB2NV21:
          //  sta = cvtColor<Tdst, ncDst, Tsrc, ncSrc, NV122BGR>(dst, &dst_fastcv);
            sta = cvtColor<NV212RGB, Tdst, ncDst, Tsrc, ncSrc>(dst, &dst_fastcv);
            EXPECT_EQ(HPC_SUCCESS, sta);
            break;
        default:
            return HPC_NOT_SUPPORTED;
    }
	if(type != BGR2NV12 && type != BGR2NV21 && type != RGB2NV21 && type != RGB2NV12)
    	dst_opencvD.download(dst_opencv);
#else
#endif



#if defined(USE_CUDA) || defined(USE_OCL)
    if(type == BGR2NV21 || type == BGR2NV12 || type == RGB2NV21 || type == RGB2NV12) {
        dst_fastcv.toHost(input_data1);
        checkResult_NV21<Tsrc, ncSrc>(input_data, input_data1, src.height(), src.width(), step/sizeof(Tsrc), diff_THR);
    } else {
        dst.toHost(output_data);
        int dst_opencv_step = dst_opencv.step;
        checkResult_colortype<Tdst, ncDst>(output_data, dst_opencv.ptr<float>(), output_height, output_width,
            dstep/sizeof(Tdst), dst_opencv_step/sizeof(float), diff_THR, type);
    }

#else

#endif

    free(input_data);
    free(input_data1);
    free(input_data_opencv);
    return 0;
}

TEST_F(halfcvtColorTest, LAB2BGR){
     imageParam p ;
    ASSERT_TRUE(!imageparams.empty());
    for(int i = 0; i < imageparams.size(); i++)
    {
        p = imageparams[i];
        testCvtColorFunc<half_t, 3, half_t, 3, LAB2BGR, 255> (p,1e-3);
        testCvtColorFunc<float, 3, float, 3, LAB2BGR, 255> (p,1e-3);
        testCvtColorFunc<half_t, 3, half_t, 4, LAB2BGR, 255> (p,1e-3);
        testCvtColorFunc<float, 3, float, 4, LAB2BGR, 255> (p,1e-3);
       
    }
   
}

TEST_F(halfcvtColorTest, BGR2LAB){
    imageParam p ;
    ASSERT_TRUE(!imageparams.empty());
    for(int i = 0; i < imageparams.size(); i++)
    {
        p = imageparams[i];
         testCvtColorFunc<half_t, 3, half_t, 3, BGR2LAB, 255> (p,3e-3);
        testCvtColorFunc<float, 3, float, 3, BGR2LAB, 255> (p,3e-3);
        testCvtColorFunc<half_t, 4, half_t, 3, BGR2LAB, 255> (p,3e-3);
        testCvtColorFunc<float, 4, float, 3, BGR2LAB, 255> (p,3e-3);
    }
      
    }
   

TEST_F(halfcvtColorTest, YCrCb2BGR){
      imageParam p ;
    ASSERT_TRUE(!imageparams.empty());
    for(int i = 0; i < imageparams.size(); i++)
    {
        p = imageparams[i];
        testCvtColorFunc<half_t, 3, half_t, 3, YCrCb2BGR, 255> (p,1e-3);
        testCvtColorFunc<float, 3, float, 3, YCrCb2BGR, 255> (p,1e-3);
        testCvtColorFunc<half_t, 3, half_t, 4, YCrCb2BGR, 255> (p,1e-3);
        testCvtColorFunc<float, 3, float, 4, YCrCb2BGR, 255> (p,1e-3);
    }
   
}
TEST_F(halfcvtColorTest, BGR2YCrCb){
      imageParam p ;
    ASSERT_TRUE(!imageparams.empty());
    for(int i = 0; i < imageparams.size(); i++)
    {
        p = imageparams[i];
         testCvtColorFunc<half_t, 3, half_t, 3, BGR2YCrCb, 255> (p,1e-3);
        testCvtColorFunc<float, 3, float, 3, BGR2YCrCb, 255> (p,1e-3);
        testCvtColorFunc<half_t, 4, half_t, 3, BGR2YCrCb, 255> (p,1e-3);
        testCvtColorFunc<float, 4, float, 3, BGR2YCrCb, 255> (p,1e-3);
       
    }
   
}
TEST_F(halfcvtColorTest, BGRA2BGR){
      imageParam p ;
    ASSERT_TRUE(!imageparams.empty());
    for(int i = 0; i < imageparams.size(); i++)
    {
        p = imageparams[i];
        testCvtColorFunc<half_t, 4, half_t, 3, BGRA2BGR, 255> (p,1e-3);
        testCvtColorFunc<float, 4, float, 3, BGRA2BGR, 255> (p,1e-3);
    }
  
}

TEST_F(halfcvtColorTest, BGR2BGRA){
      imageParam p ;
    ASSERT_TRUE(!imageparams.empty());
    for(int i = 0; i < imageparams.size(); i++)
    {
        p = imageparams[i];
         testCvtColorFunc<half_t, 3, half_t, 4, BGR2BGRA, 255> (p,1e-3);
         testCvtColorFunc<float, 3, float, 4, BGR2BGRA, 255> (p,1e-3);     
    }
   
}

TEST_F(halfcvtColorTest, BGR2RGB){
      imageParam p ;
    ASSERT_TRUE(!imageparams.empty());
    for(int i = 0; i < imageparams.size(); i++)
    {
        p = imageparams[i];
         testCvtColorFunc<half_t, 3, half_t, 3, BGR2RGB, 255> (p,1e-3);
        testCvtColorFunc<float, 3, float, 3, BGR2RGB, 255> (p,1e-3);
        testCvtColorFunc<half_t, 3, half_t, 4, BGR2RGB, 255> (p,1e-3);
        testCvtColorFunc<float, 3, float, 4, BGR2RGB, 255> (p,1e-3);
        testCvtColorFunc<half_t, 4, half_t, 3, BGR2RGB, 255> (p,1e-3);
        testCvtColorFunc<float, 4, float, 3, BGR2RGB, 255> (p,1e-3);
        testCvtColorFunc<half_t, 4, half_t, 4, BGR2RGB, 255> (p,1e-3);
        testCvtColorFunc<float, 4, float, 4, BGR2RGB, 255> (p,1e-3);

    }
   
}
TEST_F(halfcvtColorTest, BGR2GRAY){
      imageParam p ;
    ASSERT_TRUE(!imageparams.empty());
    for(int i = 0; i < imageparams.size(); i++)
    {
        p = imageparams[i];
      testCvtColorFunc<half_t, 3, half_t, 1, BGR2GRAY, 255> (p,1e-3);
        testCvtColorFunc<float, 3, float, 1, BGR2GRAY, 255> (p,1e-3);
        testCvtColorFunc<half_t, 4, half_t, 1, BGR2GRAY, 255> (p,1e-3);
        testCvtColorFunc<float, 4, float, 1, BGR2GRAY, 255> (p,1e-3);
    }
    
}

TEST_F(halfcvtColorTest, GRAY2BGR){
      imageParam p ;
    ASSERT_TRUE(!imageparams.empty());
    for(int i = 0; i < imageparams.size(); i++)
    {
        p = imageparams[i];
       testCvtColorFunc<half_t, 1, half_t, 3, GRAY2BGR, 255> (p,1e-3);
        testCvtColorFunc<float, 1, float, 3, GRAY2BGR, 255> (p,1e-3);
        testCvtColorFunc<half_t, 1, half_t, 4, GRAY2BGR, 255> (p,1e-3);
        testCvtColorFunc<float, 1, float, 4, GRAY2BGR, 255> (p,1e-3);
    }
   
}

TEST_F(halfcvtColorTest, HSV2BGR){
      imageParam p ;
    ASSERT_TRUE(!imageparams.empty());
    for(int i = 0; i < imageparams.size(); i++)
    {
        p = imageparams[i];
        testCvtColorFunc<half_t, 3, half_t, 3, HSV2BGR, 255> (p,1e-3);
        testCvtColorFunc<float, 3, float, 3, HSV2BGR, 255> (p,1e-3);
        testCvtColorFunc<half_t, 3, half_t, 4, HSV2BGR, 255> (p,1e-3);
        testCvtColorFunc<float, 3, float, 4, HSV2BGR, 255> (p,1e-3);
    }
   
}

TEST_F(halfcvtColorTest, BGR2HSV){
      imageParam p ;
    ASSERT_TRUE(!imageparams.empty());
    for(int i = 0; i < imageparams.size(); i++)
    {
        p = imageparams[i];
        testCvtColorFunc<half_t, 3, half_t, 3, BGR2HSV, 255> (p,5e-3);
        testCvtColorFunc<float, 3, float, 3, BGR2HSV, 255> (p,5e-3);
        testCvtColorFunc<half_t, 4, half_t, 3, BGR2HSV, 255> (p,5e-3);
        testCvtColorFunc<float, 4, float, 3, BGR2HSV, 255> (p,5e-3);
       
    }
    
}
