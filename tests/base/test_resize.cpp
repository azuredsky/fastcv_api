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
};


class resizeTest : public testing::Test
{
     public:
        virtual void SetUp()
        {
            std::vector<imageParam>::iterator iter = imageparams.begin();
            if(globalEnvironment::Instance()->testconfig.hasSection("resizeTest"))
            {
                Cfg::cfg_type vec_arg = globalEnvironment::Instance()->testconfig.sectionConfigVec("resizeTest");
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
                imagep={200, 190, 300,210};
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
        if(temp_diff > diff_THR)
            std::cout << "Print data12: " << i/(width * nc)<< "  " << (i%(width*nc)/nc)  << " : " << (float) data1[i] << " : " << (float) data2[i] << " : " << temp_diff << std::endl;
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
            float val3 = data1[(i + 1) * dstep + j];
            float val4 = data1[(i - 1) * dstep + j];
            float val5 = data2[(i - 1) * dstep + j];
            float val6 = data2[(i + 1) * dstep + j];

            temp_diff = fabs(val1 - val2);

#ifdef PRINT_ALL
            if (temp_diff > diff_THR) std::cout << "Print data12: " << i <<" : "
                << j << " : "<< (float) val1 << " : " << (float) val2 << std::endl;
#endif
            max = (temp_diff > max) ? temp_diff : max;
            min = (temp_diff < min) ? temp_diff : min;
        }
    }
    EXPECT_LT(max, diff_THR);
}


template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc, InterpolationType type, int val_div>
int testResizeFunc(imageParam imagep, float diff_THR) {
    size_t input_width = imagep.srcWidth;
    size_t input_height = imagep.srcHeight;
    size_t input_channels = ncSrc;
    size_t output_width = imagep.dstWidth;
    size_t output_height = imagep.dstHeight;
    size_t output_channels = ncDst;
    int spadding = imagep.spadding;
    int dpadding = imagep.dpadding;

    printf("Input image scale: width*height*channels = %ld*%ld*%ld\n", input_width, input_height, input_channels);
    printf("Output image scale: width*height*channels = %ld*%ld*%ld\n", output_width, output_height, output_channels);

    /*size_t input_data_num = input_width*input_height*input_channels;

      Tsrc* input_data = (Tsrc*)malloc(input_data_num*sizeof(Tsrc));

      randomRangeData<Tsrc, val_div>(input_data, input_data_num);*/
    size_t step = (input_width * input_channels + spadding)*sizeof(Tsrc);
    Tsrc* input_data = (Tsrc*)malloc(input_height*step);
    randomPaddingData<Tsrc, val_div>(input_data, input_height, input_width*input_channels, input_width * input_channels + spadding);

    /* std::cout << " Original data:" << std::endl;
       for(size_t i = 0; i < input_data_num; i++)
       std::cout << i << " : " << (float) input_data[i] << " " << std::endl;
     */

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

#if defined(FASTCV_USE_CUDA)
    // call opencv to verify result
    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, ncSrc), input_data, step);
    cv::Mat dst_opencv(output_height, output_width, CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst), output_data_opencv, dstep);
    cv::resize(src_opencv, dst_opencv, cv::Size(output_width, output_height), 0, 0, type);

    // call fastcv
    HPCStatus_t sta = resize<type, Tsrc, ncSrc, Tdst, ncDst, nc>(src, &dst);
    EXPECT_EQ(HPC_SUCCESS, sta);

    //x86 
    sta = resize<type, Tsrc, ncSrc, Tdst, ncDst, nc>(src_x86, &dst_x86);
    EXPECT_EQ(HPC_SUCCESS, sta);
#elif defined(FASTCV_USE_OCL)
    // call opencv to verify result
    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, ncSrc), input_data, step);
    cv::Mat dst_opencv(output_height, output_width, CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst), output_data_opencv, dstep);
    cv::resize(src_opencv, dst_opencv, cv::Size(output_width, output_height), 0, 0, type);

    // call fastcv
    HPCStatus_t sta = resize<type, Tsrc, ncSrc, Tdst, ncDst, nc>(src, &dst);
    EXPECT_EQ(HPC_SUCCESS, sta);

#else
    // call opencv to verify result
    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, ncSrc), input_data, step);
    cv::Mat dst_opencv(output_height, output_width, CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst), output_data_opencv, dstep);
    cv::resize(src_opencv, dst_opencv, cv::Size(output_width, output_height), 0, 0, type);

    // call fastcv
    HPCStatus_t sta = resize<type, Tsrc, ncSrc, Tdst, ncDst, nc>(src, &dst);
    EXPECT_EQ(HPC_SUCCESS, sta);
#endif





    // check results
#if defined(FASTCV_USE_CUDA)
    //size_t output_data_num = output_width*output_height*output_channels;
   // Tdst* dst_data_x86 = (Tdst*)malloc(output_data_num*sizeof(Tdst));
    memset(output_data,0,output_height * dstep);
    dst.toHost(output_data);
    checkResult<Tdst, ncDst>(output_data, dst_opencv.ptr<Tdst>(), output_height, output_width, dstep/sizeof(Tdst), diff_THR);
    checkResult<Tdst, ncDst>((Tdst*)dst_x86.ptr(), dst_opencv.ptr<Tdst>(), output_height, output_width, dstep/sizeof(Tdst), diff_THR);

   // free(dst_data_x86);
#elif defined(FASTCV_USE_OCL)
    //size_t output_data_num = output_width*output_height*output_channels;
    //Tdst* dst_data_x86 = (Tdst*)malloc(output_data_num*sizeof(Tdst));
    dst.toHost(output_data);
    checkResult<Tdst, ncDst>(output_data, dst_opencv.ptr<Tdst>(), output_height, output_width, dstep/sizeof(Tdst), diff_THR);
    //free(dst_data_x86);
#else
    checkResult<Tdst, ncDst>((Tdst*)dst.ptr(), dst_opencv.ptr<Tdst>(), output_height, output_width, dstep/sizeof(Tdst), diff_THR);
#endif

    return 0;
}

TEST_F(resizeTest, uchar_uchar_NEAREST_POINT){
     imageParam p ;
  //  ASSERT_TRUE(!imageparams.empty());
  //  for(int i =0; i<imageparams.size(); i++)
 //   {
        p = imagep;
        testResizeFunc<uchar, 1, uchar, 1, 1, INTERPOLATION_TYPE_NEAREST_POINT, 1>(p, 1.01);
        testResizeFunc<uchar, 3, uchar, 3, 3, INTERPOLATION_TYPE_NEAREST_POINT, 1>(p, 1.01);
        testResizeFunc<uchar, 4, uchar, 4, 4, INTERPOLATION_TYPE_NEAREST_POINT, 1>(p, 1.01);
  //  }
    
}


TEST_F(resizeTest, uchar_uchar_LINEAR){
     imageParam p ;
  //  ASSERT_TRUE(!imageparams.empty());
  //  for(int i =0; i<imageparams.size(); i++)
  //  {
        p = imagep;
        testResizeFunc<uchar, 1, uchar, 1, 1, INTERPOLATION_TYPE_LINEAR, 1>(p, 1.01);
        testResizeFunc<uchar, 3, uchar, 3, 3, INTERPOLATION_TYPE_LINEAR, 1>(p, 1.01);
        testResizeFunc<uchar, 4, uchar, 4, 4, INTERPOLATION_TYPE_LINEAR, 1>(p, 1.01);
 //   }
   
}



TEST_F(resizeTest, uchar_uchar_INTER_AREA){
   imageParam p =imagep;
   testResizeFunc<uchar, 1, uchar, 1, 1, INTERPOLATION_TYPE_INTER_AREA, 1>(p, 1.01);
   testResizeFunc<uchar, 3, uchar, 3, 3, INTERPOLATION_TYPE_INTER_AREA, 1>(p, 1.01);
   testResizeFunc<uchar, 4, uchar, 4, 4, INTERPOLATION_TYPE_INTER_AREA, 1>(p, 1.01);
}


TEST_F(resizeTest, float_float_NEAREST_POINT){
      imageParam p ;
  //  ASSERT_TRUE(!imageparams.empty());
  //  for(int i =0; i<imageparams.size(); i++)
  //  {
        p = imagep;
         testResizeFunc<float, 1, float, 1, 1, INTERPOLATION_TYPE_NEAREST_POINT, 255>(p, 1e-5);
        testResizeFunc<float, 3, float, 3, 3, INTERPOLATION_TYPE_NEAREST_POINT, 255>(p, 1e-5);
        testResizeFunc<float, 4, float, 4, 4, INTERPOLATION_TYPE_NEAREST_POINT, 255>(p, 1e-5);
   // }
  
}


TEST_F(resizeTest, float_float_LINEAR){
    imageParam p ;
 //   ASSERT_TRUE(!imageparams.empty());
//    for(int i =0; i<imageparams.size(); i++)
//    {
        p = imagep;
          testResizeFunc<float, 1, float, 1, 1, INTERPOLATION_TYPE_LINEAR, 255>(p, 3e-5);
        testResizeFunc<float, 3, float, 3, 3, INTERPOLATION_TYPE_LINEAR, 255>(p, 3e-5);
        testResizeFunc<float, 4, float, 4, 4, INTERPOLATION_TYPE_LINEAR, 255>(p, 3e-5);
  //  }
   
}



TEST_F(resizeTest, float_float_INTER_AREA){
   imageParam p = imagep;
   testResizeFunc<float, 1, float, 1, 1, INTERPOLATION_TYPE_INTER_AREA, 255>(p, 0.1);
   testResizeFunc<float, 3, float, 3, 3, INTERPOLATION_TYPE_INTER_AREA, 255>(p, 0.1);
   testResizeFunc<float, 4, float, 4, 4, INTERPOLATION_TYPE_INTER_AREA, 255>(p, 0.1);
}
 


