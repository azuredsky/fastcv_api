
//#include <gtest/gtest.h>
//#include "opencv2/opencv.hpp"
#include  "test_head.hpp"
using namespace HPC::fastcv;
struct imageParam{
    char* imageName;
};

class ReadImageTest : public testing::Test
{
    public:
	virtual void SetUp()
	{
        std::vector<imageParam>::iterator iter = imageparams.begin();
    //    Cfg testconfig("/home/liuping/testmodify/fastcv_api/tests/configure/testconfigure.txt");
      //  ASSERT_TRUE(testconfig.hasAnyConfig());
        
        std::cout<<globalEnvironment::Instance()->testconfig.m_cfg.begin()->first;
        ASSERT_TRUE(globalEnvironment::Instance()->testconfig.hasSection("ReadImageTest"));
		Cfg::cfg_type vec_readImage = globalEnvironment::Instance()->testconfig.sectionConfigVec("ReadImageTest");
		int num = vec_readImage.size();
		for(int i = 0;i < num; i++)
		{
            ASSERT_TRUE(vec_readImage[i].find("imageName") != vec_readImage[i].end());
			imagep.imageName = const_cast<char*>(vec_readImage[i]["imageName"].c_str());
            std::cout<<"imageName:  "<<imagep.imageName<<std::endl;
			iter = imageparams.insert(iter,imagep);
				
		}

		

	}
	virtual void TearDown()
	{
		if(!imageparams.empty())
			imageparams.clear();
	}
    std::vector<imageParam> imageparams;
    imageParam imagep;
};

//INSTANTIATE_TEST_CASE_P(TrueReturn, ReadImageTest, testing::ValuesIn(imageparams));


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
        std::cout << "Print data12: " << i << " : " << (float) data1[i] << " : " << (float) data2[i] << " : " << temp_diff << std::endl;
#endif		
        max = (temp_diff > max) ? temp_diff : max;
        min = (temp_diff < min) ? temp_diff : min;

    }
    EXPECT_LT(max, diff_THR);
    //ASSERT_NEAR(order, 1e-5, 1e-5);
}


template<typename T, int nc>
int testReadImageFunc(imageParam imagep, float diff_THR) {

    
#if defined(FASTCV_USE_CUDA)
    Mat<T, nc> src;
    Mat<T, nc,EcoEnv_t::ECO_ENV_X86> src_x86;
#elif defined(FASTCV_USE_OCL)
    Mat<T, nc> src;
#else
    Mat<T, nc> src;
#endif
    
    // call opencv to verify result
    cv::Mat src_opencv;
    const char *pImagename = imagep.imageName; 
    src_opencv = cv::imread(imagep.imageName);
    if(src_opencv.data == NULL)
        std::cout<<"mat read  filed!"<<std::endl;


    //printf("channel:%d\n", src_opencv.

    // call fastcv
    HPCStatus_t sta = imread<T, nc>(imagep.imageName, &src);
    EXPECT_EQ(HPC_SUCCESS, sta);


    // check results
    int output_height = src.height();
    int output_width = src.width();
    int output_data_num = output_height * output_width * nc;
    T* data_cuda = (T*)malloc(output_data_num*sizeof(T));
    src.toHost(data_cuda);
    cv::imwrite("opencv_test.bmp", src_opencv);
    sta = imwrite("fastcv_test.bmp", &src);
    EXPECT_EQ(HPC_SUCCESS, sta);
    free(data_cuda);

#if defined(FASTCV_USE_CUDA)
    HPCStatus_t sta_x86 = imread<T,nc>(imagep.imageName,&src_x86);
    EXPECT_EQ(HPC_SUCCESS, sta_x86);
    output_height = src_x86.height();
    output_width = src_x86.width();
    output_data_num = output_height * output_width * nc;
    T* data_x86 = (T*)malloc(output_data_num*sizeof(T));
    src_x86.toHost(data_x86);
   // cv::imwrite("opencv_test_x86.bmp", src_opencv);
    sta_x86 = imwrite("fastcv_test_x86.bmp", &src_x86);
    EXPECT_EQ(HPC_SUCCESS, sta_x86);
    free(data_x86);
#endif

    return 0;
}

TEST_F(ReadImageTest, uchar_3){
  //  ASSERT_TRUE(!imageparams.empty());
   // for(int i = 0;i<imageparams.size();i++)
 //   {
        const imageParam param = imagep;
        printf("imagename: %s\n",param.imageName);
        testReadImageFunc<uchar, 3>(param, 1.01);
  //  }
   // ASSERT_TRUE(globalEnvironment::Instance()->testconfig.hasSection("ReadImageTest"));


}
