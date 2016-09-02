
#include "test_head.hpp"
struct imageParam{
    char* imageName;
};

class EncodeDecodeTest : public testing::Test
{
    public:
	virtual void SetUp()
	{
        std::vector<imageParam>::iterator iter = imageparams.begin();
    //    Cfg testconfig("/home/liuping/testmodify/fastcv_api/tests/configure/testconfigure.txt");
      //  ASSERT_TRUE(testconfig.hasAnyConfig());
        
        ASSERT_TRUE(globalEnvironment::Instance()->testconfig.hasSection("EncodeDecodeTest"));
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






template<typename T, int nc>
int testEncodeDecodeFunc(imageParam imagep, float diff_THR,const char * pext) {


    
#if defined(FASTCV_USE_CUDA)
    Mat<T, nc> src;
    Mat<T, nc,EcoEnv_t::ECO_ENV_X86> src_x86;
	Mat<T, nc> de_dst;
	Mat<T, nc,EcoEnv_t::ECO_ENV_X86> de_dst_x86;
#elif defined(FASTCV_USE_OCL)
    Mat<T, nc> src;
	Mat<T, nc> de_dst;
#else
    Mat<T, nc> src;
	Mat<T, nc> de_dst;
#endif
    
    // call opencv to verify result
    cv::Mat src_opencv;
    const char *pImagename = imagep.imageName; 
    std::cout<<imagep.imageName<<std::endl;
 

    // call fastcv
    HPCStatus_t sta = imread<T, nc>(imagep.imageName, &src);
    EXPECT_EQ(HPC_SUCCESS, sta);

    std::cout<<imagep.imageName<<std::endl;
    
    const char * ext = pext;
	std::string strext(pext);
    unsigned char* buffer;
	std::vector<uchar> opencv_buffer;

    sta = imencode<T,nc>(ext,&buffer,&src);
    EXPECT_EQ(HPC_SUCCESS, sta);
//	bool opencv_ret = cv::imencode(strext,src_opencv,opencv_buffer);
   
    
    const unsigned int file_size =  buffer[0x02] +(buffer[0x03]<<8) + (buffer[0x04]<<16) + (buffer[0x05]<<24);
    sta = imdecode<T,nc>(buffer, file_size, &de_dst);
    EXPECT_EQ(HPC_SUCCESS, sta);
	
	sta = imwrite("fastcv_test.bmp", &de_dst);
	 EXPECT_EQ(HPC_SUCCESS, sta);


    return 0;
}

TEST_F(EncodeDecodeTest, bmp){
  //  ASSERT_TRUE(!imageparams.empty());
   // for(int i = 0;i<imageparams.size();i++)
 //   {
        const imageParam param = imagep;
		const char * ext = "bmp";
        printf("imagename: %s\n",param.imageName);
        testEncodeDecodeFunc<uchar, 3>(param, 1.01, ext);
  //  }
   // ASSERT_TRUE(globalEnvironment::Instance()->testconfig.hasSection("ReadImageTest"));


}

TEST_F(EncodeDecodeTest, png){
  //  ASSERT_TRUE(!imageparams.empty());
   // for(int i = 0;i<imageparams.size();i++)
 //   {
        const imageParam param = imagep;
		const char * ext = "png";
        printf("imagename: %s\n",param.imageName);
        testEncodeDecodeFunc<uchar, 3>(param, 1.01,ext);
  //  }
   // ASSERT_TRUE(globalEnvironment::Instance()->testconfig.hasSection("ReadImageTest"));


}

TEST_F(EncodeDecodeTest, jpeg){
  //  ASSERT_TRUE(!imageparams.empty());
   // for(int i = 0;i<imageparams.size();i++)
 //   {
        const imageParam param = imagep;
		const char * ext = "jpeg";
        printf("imagename: %s\n",param.imageName);
        testEncodeDecodeFunc<uchar, 3>(param, 1.01, ext);
  //  }
   // ASSERT_TRUE(globalEnvironment::Instance()->testconfig.hasSection("ReadImageTest"));


}
