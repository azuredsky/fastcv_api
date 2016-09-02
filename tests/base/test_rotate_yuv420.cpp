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
    float degree;
};


class rotateYUV420Test : public testing::Test
{
    public:
        virtual void SetUp()
        {
            std::vector<imageParam>::iterator iter = imageparams.begin();
            if(globalEnvironment::Instance()->testconfig.hasSection("rotateYUV420Test"))
            {
                Cfg::cfg_type vec_arg = globalEnvironment::Instance()->testconfig.sectionConfigVec("rotateYUV420Test");
                int num = vec_arg.size();
                for(int i = 0;i < num; i++)
                {
                    ASSERT_TRUE(vec_arg[i].find("srcHeight")!=vec_arg[i].end())<<"argument:srcHeight not found";
                    ASSERT_TRUE(vec_arg[i].find("srcWidth")!=vec_arg[i].end())<<"argument:srcWidth not found";
                    ASSERT_TRUE(vec_arg[i].find("dstWidth")!=vec_arg[i].end())<<"argument:dstWidth not found";
                    ASSERT_TRUE(vec_arg[i].find("dstHeight")!=vec_arg[i].end())<<"argument:dstHeight not found";                   
                    ASSERT_TRUE(vec_arg[i].find("spadding")!=vec_arg[i].end())<<"argument:spadding not found";
                    ASSERT_TRUE(vec_arg[i].find("dpadding")!=vec_arg[i].end())<<"argument:dpadding not found";
                     ASSERT_TRUE(vec_arg[i].find("degree")!=vec_arg[i].end())<<"argument:degree not found";
                    imagep.srcHeight = std::stoul(vec_arg[i]["srcHeight"]);
                    imagep.srcWidth = std::stoul(vec_arg[i]["srcWidth"]);
                    imagep.dstWidth = std::stoul(vec_arg[i]["dstWidth"]);
                    imagep.dstHeight = std::stoul(vec_arg[i]["dstHeight"]);              
                    imagep.spadding = std::stoi(vec_arg[i]["spadding"]);
                    imagep.dpadding = std::stoi(vec_arg[i]["dpadding"]);
                     imagep.degree = std::stof(vec_arg[i]["degree"]);
                    imageparams.push_back(imagep);

                }
            }
            else
            {
                imagep={200,200,200,200, 0, 0, 270};
                 imageparams.push_back(imagep);
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
            //printf("%d\n",data[i]);
        }
    }
}

template<typename T, int nc>
void checkResult(const T *data1, const T *data2, const int height, const int width, int sstep, int dstep, const float diff_THR) {
    double max = DBL_MIN;
    double min = DBL_MAX;
    double temp_diff = 0.0;

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width*nc; j++){
            float val1 = data1[i * sstep +j];
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
template<YUV420Type yt, typename T, int ncSrc, int ncDst, int nc, int val_div>
int testRotateYUV420Func(imageParam imagep, float diff_THR) {
    size_t input_width = imagep.srcWidth;
    size_t input_height = imagep.srcHeight*3/2;
    size_t input_channels = ncSrc;
    size_t output_width = imagep.dstWidth;
    size_t output_height = imagep.dstHeight*3/2;
    size_t output_channels = ncDst;
    int spadding = imagep.spadding;
    int dpadding = imagep.dpadding;
    float degree = imagep.degree;

    printf("Input image scale: width*height*channels = %ld*%ld*%ld\n", input_width, input_height, input_channels);
    printf("Output image scale: width*height*channels = %ld*%ld*%ld\n", output_width, output_height, output_channels);

    int inStride = (input_width * input_channels + spadding);
    size_t step = inStride * sizeof(T);
    T* input_data = (T*)malloc(input_height*step);
    T* input_data_tmp = (T*)malloc(input_height*step);
    randomPaddingData<T, val_div>(input_data, input_height, input_width*input_channels, input_width * input_channels + spadding);
    randomPaddingData<T, val_div>(input_data_tmp, input_height, input_width*input_channels, input_width * input_channels + spadding);

#if defined(FASTCV_USE_CUDA)
    Mat<T, ncSrc> src(input_height, input_width, step);
    Mat<T, ncSrc> src_tmp(input_height, input_width, step);
    src.fromHost(input_data);

    const Mat<T, ncSrc,EcoEnv_t::ECO_ENV_X86> src_x86(input_height, input_width, input_data, step);
    Mat<T, ncSrc,EcoEnv_t::ECO_ENV_X86> src_tmp_x86(input_height, input_width, input_data_tmp, step);
#elif defined(FASTCV_USE_OCL)
    Mat<T, ncSrc> src(input_height, input_width, step);
    Mat<T, ncSrc> src_tmp(input_height, input_width, input_data_tmp, step);
    src.fromHost(input_data);
#else
    const Mat<T, ncSrc> src(input_height, input_width, input_data, step);
    Mat<T, ncSrc> src_tmp(input_height, input_width, input_data_tmp, step);
#endif

    size_t dstep = (output_width * output_channels + dpadding)*sizeof(T);
    T* output_data = (T* )malloc(output_height*dstep);
    T* output_data_tmp = (T* )malloc(output_height*dstep);

#if defined(FASTCV_USE_CUDA)
    Mat<T, ncDst> dst(output_height, output_width, dstep);
    Mat<T, ncDst> dst_tmp(output_height, output_width, dstep);

    Mat<T, ncDst,EcoEnv_t::ECO_ENV_X86> dst_x86(output_height, output_width, output_data, dstep);
    Mat<T, ncDst,EcoEnv_t::ECO_ENV_X86> dst_tmp_x86(output_height, output_width, output_data_tmp, dstep);
#elif defined(FASTCV_USE_OCL)
    Mat<T, ncDst> dst(output_height, output_width, dstep);
    Mat<T, ncDst> dst_tmp(output_height, output_width, output_data_tmp, dstep);
#else
    Mat<T, ncDst> dst(output_height, output_width, output_data, dstep);
    Mat<T, ncDst> dst_tmp(output_height, output_width, output_data_tmp, dstep);
#endif

    //call fastcv verify result
    if(fabs(degree - 90) < 1e-9 || fabs(degree - 270) < 1e-9){
        HPCStatus_t sta = rotate_YUV420<yt, T>(src, &dst_tmp, degree);
        sta = rotate_YUV420<yt, T>(dst_tmp, &src_tmp, degree);
        sta = rotate_YUV420<yt, T>(src_tmp, &dst_tmp, degree);
        sta = rotate_YUV420<yt, T>(dst_tmp, &src_tmp, degree);
        EXPECT_EQ(HPC_SUCCESS, sta);
    }
    if(fabs(degree - 180) < 1e-9){
        HPCStatus_t sta = rotate_YUV420<yt, T>(src, &dst_tmp, degree);
        sta = rotate_YUV420<yt, T>(dst_tmp, &src_tmp, degree);
        EXPECT_EQ(HPC_SUCCESS, sta);
    }

#if defined(FASTCV_USE_CUDA)   //test x86 in cuda test
    //call fastcv verify result
    if(fabs(degree - 90) < 1e-9 || fabs(degree - 270) < 1e-9){
        HPCStatus_t sta = rotate_YUV420<yt, T>(src_x86, &dst_tmp_x86, degree);
        sta = rotate_YUV420<yt, T>(dst_tmp_x86, &src_tmp_x86, degree);
        sta = rotate_YUV420<yt, T>(src_tmp_x86, &dst_tmp_x86, degree);
        sta = rotate_YUV420<yt, T>(dst_tmp_x86, &src_tmp_x86, degree);
        EXPECT_EQ(HPC_SUCCESS, sta);
    }
    if(fabs(degree - 180) < 1e-9){
        HPCStatus_t sta = rotate_YUV420<yt, T>(src_x86, &dst_tmp_x86, degree);
        sta = rotate_YUV420<yt, T>(dst_tmp_x86, &src_tmp_x86, degree);
        EXPECT_EQ(HPC_SUCCESS, sta);
    }
#endif
#if defined(FASTCV_USE_CUDA)
    src.toHost(input_data);
    src_tmp.toHost(input_data_tmp);
    checkResult<T, ncDst>(input_data, input_data_tmp, input_height, input_width, step/sizeof(T), step/sizeof(T), diff_THR);

    checkResult<T, ncDst>((T *)src_x86.ptr(), input_data_tmp, input_height, input_width, step/sizeof(T), step/sizeof(T), diff_THR);

#elif defined(FASTCV_USE_OCL)

#else
     // check results

    checkResult<T, ncDst>((T *)src.ptr(), input_data_tmp, input_height, input_width, step/sizeof(T), step/sizeof(T), diff_THR);
#endif



    free(input_data);
    free(input_data_tmp);
    free(output_data);
    free(output_data_tmp);
    return 0;
}

TEST_F(rotateYUV420Test, uchar_uchar_YUV420){
    imageParam p;
   // ASSERT_TRUE(!imageparams.empty());

  //  for(int i = 0;i<imageparams.size();i++)
  //  {
        p = imagep;
         testRotateYUV420Func<YUV420_NV21, uchar, 1, 1, 1, 1>(p, 1.01);
        testRotateYUV420Func<YUV420_NV12, uchar, 1, 1, 1, 1>(p, 1.01);
        testRotateYUV420Func<YUV420_I420, uchar, 1, 1, 1, 1>(p, 1.01);
  //  }
  
}

