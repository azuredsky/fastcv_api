#include "test_head.hpp"
using namespace HPC::fastcv;

//#define PRINT_ALL
struct imageParam{
        size_t srcWidth;
        size_t srcHeight;
        size_t dstWidth;
        size_t dstHeight;
};

class remapTest : public testing::Test
{
         public:
        virtual void SetUp()
        {
            std::vector<imageParam>::iterator iter = imageparams.begin();
            if(globalEnvironment::Instance()->testconfig.hasSection("remapTest"))
            {
                Cfg::cfg_type vec_arg = globalEnvironment::Instance()->testconfig.sectionConfigVec("remapTest");
                int num = vec_arg.size();
                for(int i = 0;i < num; i++)
                {
                    ASSERT_TRUE(vec_arg[i].find("srcHeight")!=vec_arg[i].end())<<"argument:srcHeight not found";
                    ASSERT_TRUE(vec_arg[i].find("srcWidth")!=vec_arg[i].end())<<"argument:srcWidth not found";
                    ASSERT_TRUE(vec_arg[i].find("dstWidth")!=vec_arg[i].end())<<"argument:dstWidth not found";
                    ASSERT_TRUE(vec_arg[i].find("dstHeight")!=vec_arg[i].end())<<"argument:dstHeight not found";                   
                  
                    imagep.srcHeight = std::stoul(vec_arg[i]["srcHeight"]);
                    imagep.srcWidth = std::stoul(vec_arg[i]["srcWidth"]);
                    imagep.dstWidth = std::stoul(vec_arg[i]["dstWidth"]);
                    imagep.dstHeight = std::stoul(vec_arg[i]["dstHeight"]);              
                  
                    iter = imageparams.insert(iter,imagep);

                }
            }
            else
            {
                 imagep={640, 480, 640 , 480};
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
void randomRangeData(T *data, const size_t num,int maxNum =255){
        size_t tmp;
        clock_t ct = clock();
        srand((unsigned int)ct);

        for(size_t i = 0; i < num; i++){
                tmp = rand()% maxNum;
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
void checkResult(T *data1, T *data2, int height, int width, float diff_THR) {
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



/*
int main(int argc, char *argv[]){
        if(argc != 1 && argc != 5){
                printf("Error params!\n");
                printf("Usage: ./run src_width src_height \
                                dst_width dst_height\n");
        }
        imagep={640, 480, 640 , 480};
      //  imagep={4888,4888,2222,2262};
        if(argc == 5){
                imagep.srcWidth = atoi(argv[1]);
                imagep.srcHeight = atoi(argv[2]);
                imagep.dstWidth = atoi(argv[3]);
                imagep.dstHeight = atoi(argv[4]);
        }

        testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
}

*/
template<typename Tsrc, int ncSrc, typename Tdst, int ncDst, int nc, InterpolationType type, int val_div>
int testRemapFunc(imageParam imagep, float diff_THR) {
        size_t input_width = imagep.srcWidth;
        size_t input_height = imagep.srcHeight;
        size_t input_channels = ncSrc;
        size_t output_width = imagep.dstWidth;
        size_t output_height = imagep.dstHeight;
        size_t output_channels = ncDst;

        printf("Input image scale: width*height*channels = %ld*%ld*%ld\n", input_width, input_height, input_channels);
        printf("Output image scale: width*height*channels = %ld*%ld*%ld\n", output_width, output_height, output_channels);

        size_t input_data_num = input_width*input_height*input_channels;
        size_t output_data_num = output_width*output_height*output_channels;
        size_t map_num = output_width*output_height;
        Tsrc* input_data = (Tsrc*)malloc(input_data_num*sizeof(Tsrc));

        randomRangeData<Tsrc, val_div>(input_data, input_data_num,255);

        float* tempX = (float *)malloc(map_num * sizeof(float));
        float* tempY = (float *)malloc(map_num * sizeof(float));
        randomRangeData<float, 1>(tempX, map_num, input_width);
        randomRangeData<float, 1>(tempY, map_num, input_height);

        //cv::Mat inv_mat(2, 3, CV_32FC1, inv_warpMat);
   /*      std::cout << " Original data:" << std::endl;
           for(size_t i = 0; i < input_data_num; i++)
           std::cout << i << " : " << (float) input_data[i] << "\t " << (float)temp[i]<< std::endl;
     */
   		 struct timeval start, end;
		double duration;
#if defined(FASTCV_USE_CUDA) 
        Mat<Tsrc, ncSrc> src(input_height, input_width);
        src.fromHost(input_data);
        Mat<float, 1> mapX(output_height, output_width);
        mapX.fromHost(tempX);
        Mat<float, 1> mapY(output_height, output_width);
        mapY.fromHost(tempY);

        //X86
        const Mat<Tsrc, ncSrc,EcoEnv_t::ECO_ENV_X86> src_x86(input_height, input_width, input_data);
        Mat<float, 1,EcoEnv_t::ECO_ENV_X86> mapX_x86(output_height, output_width, tempX);
        Mat<float, 1,EcoEnv_t::ECO_ENV_X86> mapY_x86(output_height, output_width, tempY);
#elif defined(FASTCV_USE_OCL)
        Mat<Tsrc, ncSrc> src(input_height, input_width);
        src.fromHost(input_data);
        Mat<float, 1> mapX(output_height, output_width);
        mapX.fromHost(tempX);
        Mat<float, 1> mapY(output_height, output_width);
        mapY.fromHost(tempY);

#else
        const Mat<Tsrc, ncSrc> src(input_height, input_width, input_data);
        Mat<float, 1> mapX(output_height, output_width, tempX);
        Mat<float, 1> mapY(output_height, output_width, tempY);
#endif

#if defined(FASTCV_USE_CUDA)
        // call opencv to verify result
        cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, ncSrc), input_data);
        cv::Mat dst_opencv(output_height, output_width, CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst));
        cv::Mat opencvMapX(output_height, output_width, CV_MAKETYPE(cv::DataType<float>::depth, 1), tempX);
        cv::Mat opencvMapY(output_height, output_width, CV_MAKETYPE(cv::DataType<float>::depth, 1), tempY);
        cv::remap(src_opencv, dst_opencv,opencvMapX,opencvMapY,CV_INTER_LINEAR,BORDER_CONSTANT);

        // call fastcv
        Mat<Tdst, ncDst> dst(output_height, output_width);
        zeros<Tdst, ncDst>(&dst);
		for(int i=0;i<10;i++){
			gettimeofday(&start, NULL);
		    HPCStatus_t sta = remap<Tsrc, ncSrc, ncDst, nc>(src, mapX,mapY, type, &dst);
			gettimeofday(&end, NULL);
			duration = timingExec(start, end);
			printf("%d\tremap  :%f ms\n",i,duration);
        	EXPECT_EQ(HPC_SUCCESS, sta);
		}
        //x86
         Mat<Tdst, ncDst,EcoEnv_t::ECO_ENV_X86> dst_x86(output_height, output_width);
         zeros<Tdst, ncDst,EcoEnv_t::ECO_ENV_X86>(&dst_x86);
		for(int i=0;i<10;i++){
			gettimeofday(&start, NULL);
		       HPCStatus_t sta = remap<Tsrc, ncSrc, ncDst, nc>(src_x86, mapX_x86,mapY_x86, type, &dst_x86);
			gettimeofday(&end, NULL);
			duration = timingExec(start, end);
		//	printf("%d\tfastcv_remap  :%f ms\n",i,duration);
        	EXPECT_EQ(HPC_SUCCESS, sta);
		}
#elif defined(FASTCV_USE_OCL)
        // call opencv to verify result
        cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, ncSrc), input_data);
        cv::Mat dst_opencv(output_height, output_width, CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst));
        cv::Mat opencvMapX(output_height, output_width, CV_MAKETYPE(cv::DataType<float>::depth, 1), tempX);
        cv::Mat opencvMapY(output_height, output_width, CV_MAKETYPE(cv::DataType<float>::depth, 1), tempY);
        cv::remap(src_opencv, dst_opencv,opencvMapX,opencvMapY,CV_INTER_LINEAR,BORDER_CONSTANT);

        // call fastcv
        Mat<Tdst, ncDst> dst(output_height, output_width);
        zeros<Tdst, ncDst>(&dst);
		for(int i=0;i<10;i++){
			gettimeofday(&start, NULL);
		    HPCStatus_t sta = remap<Tsrc, ncSrc, ncDst, nc>(src, mapX,mapY, type, &dst);
			gettimeofday(&end, NULL);
			duration = timingExec(start, end);
			printf("%d\tremap  :%f ms\n",i,duration);
        	EXPECT_EQ(HPC_SUCCESS, sta);
		}
#else
        // call opencv to verify result
        cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, ncSrc), input_data);
        cv::Mat dst_opencv(output_height, output_width, CV_MAKETYPE(cv::DataType<Tdst>::depth, ncDst));
        cv::Mat opencvMapX(output_height, output_width, CV_MAKETYPE(cv::DataType<float>::depth, 1), tempX);
        cv::Mat opencvMapY(output_height, output_width, CV_MAKETYPE(cv::DataType<float>::depth, 1), tempY);
        cv::remap(src_opencv, dst_opencv,opencvMapX,opencvMapY,CV_INTER_LINEAR,BORDER_CONSTANT);
    	for(int i=0;i<10;i++){
			gettimeofday(&start, NULL);
            cv::remap(src_opencv, dst_opencv,opencvMapX,opencvMapY,CV_INTER_LINEAR,BORDER_CONSTANT);
			gettimeofday(&end, NULL);
			duration = timingExec(start, end);
		//	printf("%d\topencv_remap  :%f ms\n",i,duration);
		}
        // call fastcv
        Mat<Tdst, ncDst> dst(output_height, output_width);
        zeros<Tdst, ncDst>(&dst);
		for(int i=0;i<10;i++){
			gettimeofday(&start, NULL);
		    HPCStatus_t sta = remap<Tsrc, ncSrc, ncDst, nc>(src, mapX,mapY, type, &dst);
			gettimeofday(&end, NULL);
			duration = timingExec(start, end);
		//	printf("%d\tfastcv_remap  :%f ms\n",i,duration);
        	EXPECT_EQ(HPC_SUCCESS, sta);
		}
#endif

        // check results
#if defined(FASTCV_USE_CUDA)
        //size_t output_data_num = output_width*output_height*output_channels;
        Tdst* dst_data_x86 = (Tdst*)malloc(output_data_num*sizeof(Tdst));
        dst.toHost(dst_data_x86);
        checkResult<Tdst, ncDst>(dst_data_x86, dst_opencv.ptr<Tdst>(), output_height, output_width, diff_THR);
        free(dst_data_x86);

        checkResult<Tdst, ncDst>((Tdst*)dst_x86.ptr(), dst_opencv.ptr<Tdst>(), output_height, output_width, diff_THR);

#elif defined(FASTCV_USE_OCL)
        Tdst* dst_data_x86 = (Tdst*)malloc(output_data_num*sizeof(Tdst));
        dst.toHost(dst_data_x86);
        checkResult<Tdst, ncDst>(dst_data_x86, dst_opencv.ptr<Tdst>(), output_height, output_width, diff_THR);
        free(dst_data_x86);
#else
        checkResult<Tdst, ncDst>((Tdst *)dst.ptr(), dst_opencv.ptr<Tdst>(), output_height, output_width, diff_THR);
#endif

		free(tempX);
		free(tempY);
        return 0;
}
/*
TEST_P(RemapParamTest, uchar_uchar_NEAREST_POINT){
        imageParam p = GetParam();
        testRemapFunc<uchar, 1, uchar, 1, 1, INTERPOLATION_TYPE_NEAREST_POINT, 1>(p, 1.01);
        testRemapFunc<uchar, 3, uchar, 3, 3, INTERPOLATION_TYPE_NEAREST_POINT, 1>(p, 1.01);
        testRemapFunc<uchar, 4, uchar, 4, 4, INTERPOLATION_TYPE_NEAREST_POINT, 1>(p, 1.01);
}
*/

TEST_F(remapTest, uchar_uchar_LINEAR){
    imageParam p ;
  //  ASSERT_TRUE(!imageparams.empty());
  //  for(int i =0; i<imageparams.size(); i++)
 //   {
        p = imagep;
        testRemapFunc<uchar, 1, uchar, 1, 1, INTERPOLATION_TYPE_LINEAR, 1>(p, 1.01);
        testRemapFunc<uchar, 3, uchar, 3, 3, INTERPOLATION_TYPE_LINEAR, 1>(p, 1.01);
        testRemapFunc<uchar, 4, uchar, 4, 4, INTERPOLATION_TYPE_LINEAR, 1>(p, 1.01);
        
  //  }
    

}
/*
   TEST_P(RemapParamTest, float_float_NEAREST_POINT){
   imageParam p = GetParam();
   testRemapFunc<float, 1, float, 1, 1, INTERPOLATION_TYPE_NEAREST_POINT, 255>(p, 1e-5);
   testRemapFunc<float, 3, float, 3, 3, INTERPOLATION_TYPE_NEAREST_POINT, 255>(p, 1e-5);
   testRemapFunc<float, 4, float, 4, 4, INTERPOLATION_TYPE_NEAREST_POINT, 255>(p, 1e-5);
   }
   */


TEST_F(remapTest, float_float_LINEAR){
     imageParam p ;
  //  ASSERT_TRUE(!imageparams.empty());
  //  for(int i =0; i<imageparams.size(); i++)
 //   {
        p = imagep;
         testRemapFunc<float, 1, float, 1, 1, INTERPOLATION_TYPE_LINEAR, 255>(p, 1e-5);
        testRemapFunc<float, 3, float, 3, 3, INTERPOLATION_TYPE_LINEAR, 255>(p, 1e-5);
        testRemapFunc<float, 4, float, 4, 4, INTERPOLATION_TYPE_LINEAR, 255>(p, 1e-5);
        
  //  }
  

}

