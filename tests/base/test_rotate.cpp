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


class rotateTest : public testing::Test
{
    public:
        virtual void SetUp()
        {
            imageparams.clear();
            std::vector<imageParam>(imageparams).swap(imageparams);
            std::vector<imageParam>::iterator iter = imageparams.begin();
            if(globalEnvironment::Instance()->testconfig.hasSection("rotateTest"))
            {
                Cfg::cfg_type vec_arg = globalEnvironment::Instance()->testconfig.sectionConfigVec("rotateTest");
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
                     std::cout<<"size:"<<std::endl;
                     imageparams.push_back(imagep);
                     std::cout<<"size:"<<std::endl;
                     iter++;

                }
            }
            else
            {
                //imagep={200,200,200,200, 0, 0, 270};
                imagep.srcHeight = 200;
                imagep.srcWidth = 200;
                imagep.dstWidth = 200;
                imagep.dstHeight = 200;
                imagep.spadding = 0;
                imagep.dpadding = 0;
                imagep.degree = 270;
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



template<typename T, int nc>
static void bgrRotate90degreeRaw(int inHeight, int inWidth, int inWidthStride, const T* inData,
        int outHeight, int outWidth, int outWidthStride, T* outData){
    for(int i = 0; i < outHeight; i++){
        for(int j = 0; j < outWidth; j++){
            for(int c = 0; c < nc; c++){
                outData[i * outWidthStride + j * nc + c] = inData[(inHeight - j - 1) * inWidthStride + i * nc + c];
            }
        }
    }
}
template<typename T, int nc>
static void bgrRotate180degreeRaw(int inHeight, int inWidth, int inWidthStride, const T* inData,
        int outHeight, int outWidth, int outWidthStride, T* outData){
    for(int i = 0; i < outHeight; i++){
        for(int j = 0; j < outWidth; j++){
            for(int c = 0; c < nc; c++){
                outData[i * outWidthStride + j * nc + c] = inData[(inHeight - i - 1) * inWidthStride + (inWidth - j - 1) * nc + c];
            }
        }
    }
}
template<typename T, int nc>
static void bgrRotate270degreeRaw(int inHeight, int inWidth, int inWidthStride, const T* inData,
        int outHeight, int outWidth, int outWidthStride, T* outData){
    for(int i = 0; i < outHeight; i++){
        for(int j = 0; j < outWidth; j++){
            for(int c = 0; c < nc; c++){
                outData[i * outWidthStride + j * nc + c] = inData[j * inWidthStride + (inWidth - i - 1) * nc + c];;
            }
        }
    }
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
            //printf("%d\n",data[i]);
        }
    }
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
            std::cout << "Print data12: " << i << " : " <<  j << " : "<< (float) val1 << " : " << (float) val2 << " : " << temp_diff << std::endl;
#endif
            max = (temp_diff > max) ? temp_diff : max;
            min = (temp_diff < min) ? temp_diff : min;
        }
    }
    EXPECT_LT(max, diff_THR);
}


template<typename T, int ncSrc, int ncDst, int nc, int val_div>
int testRotateFunc(imageParam imagep, float diff_THR) {
    size_t input_width = imagep.srcWidth;
    size_t input_height = imagep.srcHeight;
    size_t input_channels = ncSrc;
    size_t output_width = imagep.dstWidth;
    size_t output_height = imagep.dstHeight;
    size_t output_channels = ncDst;
    int spadding = imagep.spadding;
    int dpadding = imagep.dpadding;
    float degree = imagep.degree;

    printf("Input image scale: width*height*channels = %ld*%ld*%ld\n", input_width, input_height, input_channels);
    printf("Output image scale: width*height*channels = %ld*%ld*%ld\n", output_width, output_height, output_channels);

    size_t step = (input_width * input_channels + spadding)*sizeof(T);
    T* input_data = (T*)malloc(input_height*step);
    randomPaddingData<T, val_div>(input_data, input_height, input_width*input_channels, input_width * input_channels + spadding);

#if defined(FASTCV_USE_CUDA)
    Mat<T, ncSrc> src(input_height, input_width, step);
    src.fromHost(input_data);
    const Mat<T, ncSrc,EcoEnv_t::ECO_ENV_X86> src_x86(input_height,input_width,input_data,step);
#elif defined(FASTCV_USE_OCL)
    Mat<T, ncSrc> src(input_height, input_width, step);
    src.fromHost(input_data);
#else
    const Mat<T, ncSrc> src(input_height, input_width, input_data, step);
#endif

    size_t dstep = (output_width * output_channels + dpadding)*sizeof(T);
    T* output_data = (T* )malloc(output_height*dstep);
    T* output_data_raw = (T* )malloc(output_height*dstep);
    randomPaddingData<T, val_div>(output_data, output_height,
            output_width*output_channels, output_width * output_channels + dpadding);

    randomPaddingData<T, val_div>(output_data_raw, output_height,
        output_width*output_channels, output_width * output_channels + dpadding);


#if defined(FASTCV_USE_CUDA)
    Mat<T, ncDst> dst(output_height, output_width, dstep);
    dst.fromHost(output_data);

    Mat<T,ncDst,EcoEnv_t::ECO_ENV_X86> dst_x86(output_height,output_width,output_data,dstep);
#elif defined(FASTCV_USE_OCL)
    Mat<T, ncDst> dst(output_height, output_width, dstep);
    dst.fromHost(output_data);
#else
    Mat<T, ncDst> dst(output_height, output_width, output_data, dstep);
#endif

    // call fastcv
    HPCStatus_t sta = rotate<T, ncSrc, ncDst, nc>(src, &dst, degree);
    EXPECT_EQ(HPC_SUCCESS, sta);

#if defined(FASTCV_USE_CUDA)
    HPCStatus_t sta_x86= rotate<T,ncSrc,ncDst,nc>(src_x86,&dst_x86,degree);
    EXPECT_EQ(HPC_SUCCESS,sta_x86);
#endif
    // call raw function to verify result
    if(fabs(degree - 90) < 1e-9){
        bgrRotate90degreeRaw<T, nc>(input_height, input_width, step/sizeof(T), input_data,
            output_height, output_width, dstep/sizeof(T), output_data_raw);
    }
    if(fabs(degree - 180) < 1e-9)
        bgrRotate180degreeRaw<T, nc>(input_height, input_width, step/sizeof(T), input_data,
            output_height, output_width, dstep/sizeof(T), output_data_raw);
    if(fabs(degree - 270) < 1e-9)
        bgrRotate270degreeRaw<T, nc>(input_height, input_width, step/sizeof(T), input_data,
            output_height, output_width, dstep/sizeof(T), output_data_raw);


    // check results
#if defined(FASTCV_USE_CUDA)
    dst.toHost(output_data);
	checkResult<T, ncDst>(output_data, output_data_raw, output_height, output_width, dstep/sizeof(T), diff_THR);
    checkResult<T, ncDst>((T *)dst_x86.ptr(), output_data_raw, output_height, output_width, dstep/sizeof(T), diff_THR);
#else 

    checkResult<T, ncDst>((T *)dst.ptr(), output_data_raw, output_height, output_width, dstep/sizeof(T), diff_THR);

#endif
    free(input_data);
    free(output_data_raw);
    free(output_data);
    return 0;
}
TEST_F(rotateTest, uchar_uchar_Standard){
    imageParam p;
 //   ASSERT_TRUE(!imageparams.empty());

  //  for(int i = 0;i<imageparams.size();i++)
 //  {
        p = imagep;
       testRotateFunc<uchar, 1, 1, 1, 1>(p, 1.01);
        testRotateFunc<uchar, 3, 3, 3, 1>(p, 1.01);
        testRotateFunc<uchar, 4, 4, 4, 1>(p, 1.01);

 //   }
    
}

TEST_F(rotateTest, float_float_Standard){
    imageParam p;
   // ASSERT_TRUE(!imageparams.empty());

  //  for(int i = 0;i<imageparams.size();i++)
  //  {
        p = imagep;
       testRotateFunc<float, 1, 1, 1, 255>(p, 1e-5);
        testRotateFunc<float, 3, 3, 3, 255>(p, 1e-5);
        testRotateFunc<float, 4, 4, 4, 255>(p, 1e-5);
 //   }
   
}
