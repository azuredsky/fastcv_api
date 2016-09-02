#include "test_head.hpp"

using namespace HPC::fastcv;

struct imageParam{
    size_t srcWidth;
    size_t srcHeight;
    size_t dstWidth;
    size_t dstHeight;
    int spadding;
   
};


class pixeldereferTest : public testing::Test
{
    public:
        virtual void SetUp()
        {
            std::vector<imageParam>::iterator iter = imageparams.begin();
            if(globalEnvironment::Instance()->testconfig.hasSection("pixeldereferTest"))
            {
                Cfg::cfg_type vec_arg = globalEnvironment::Instance()->testconfig.sectionConfigVec("pixeldereferTest");
                int num = vec_arg.size();
                for(int i = 0;i < num; i++)
                {
                    ASSERT_TRUE(vec_arg[i].find("srcHeight")!=vec_arg[i].end())<<"argument:srcHeight not found";
                    ASSERT_TRUE(vec_arg[i].find("srcWidth")!=vec_arg[i].end())<<"argument:srcWidth not found";
                    ASSERT_TRUE(vec_arg[i].find("dstWidth")!=vec_arg[i].end())<<"argument:dstWidth not found";
                    ASSERT_TRUE(vec_arg[i].find("dstHeight")!=vec_arg[i].end())<<"argument:dstHeight not found";                   
                    ASSERT_TRUE(vec_arg[i].find("spadding")!=vec_arg[i].end())<<"argument:spadding not found";
                  
                    imagep.srcHeight = std::stoul(vec_arg[i]["srcHeight"]);
                    imagep.srcWidth = std::stoul(vec_arg[i]["srcWidth"]);
                    imagep.dstWidth = std::stoul(vec_arg[i]["dstWidth"]);
                    imagep.dstHeight = std::stoul(vec_arg[i]["dstHeight"]);              
                    imagep.spadding = std::stoi(vec_arg[i]["spadding"]);
                  
                    iter = imageparams.insert(iter,imagep);

                }
            }
            else
            {
                imagep={200,200, 200, 200, 3};
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
static void checkResult(const T *data1, const T *data2, const int height, const int width, int sstep, int dstep) {
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
    if((max) > 1e-5)
        printf("error too large\n");
    else{
        printf("test passed\n");
    }
}

#define T uchar
#define nc 1
int testPixelderefer(imageParam imagep)
{
    size_t input_width = imagep.srcWidth;
    size_t input_height = imagep.srcHeight;
    size_t output_width = imagep.dstWidth;
    size_t output_height = imagep.dstHeight;
    int spadding = imagep.spadding;
    
    printf("Input image scale: width*height*channels = %ld*%ld*%d\n",
            input_width, input_height, nc);
    printf("Output image scale: width*height*channels = %ld*%ld*%d\n",
            output_width, output_height, nc);

    size_t step = (input_width * nc + spadding)*sizeof(T);
    T* input_data = (T *)malloc(input_height*step);
    randomPaddingData<T, 1>(input_data, input_height, input_width*nc, input_width * nc + spadding);

    Mat<T, nc> src(input_height, input_width, input_data, step);

    size_t dstep = output_width*nc*sizeof(T);
    T* output_data = (T *)malloc(output_height * dstep);
    randomPaddingData<T, 1>(output_data, output_height,
            output_width*nc, output_width * nc);

    Mat<T, nc> dst(output_height, output_width, output_data, dstep);

    //get pixel and signed tp dst
    int i, j, k = 0;
    for(i = 0; i < input_height; i++){
        for(j = 0; j < input_width; j++){
            if(sizeof(T) == 1 && nc == 1){
                uchar pixel = src.at<uchar>(i, j);
                output_data[k++] = pixel;
            } else if(sizeof(T) == 1 && nc == 3){
                Vec3b pixel = src.at<Vec3b>(i, j);
                output_data[k++] = pixel[0];
                output_data[k++] = pixel[1];
                output_data[k++] = pixel[2];
            } else if(sizeof(T) == 1 && nc == 4){
                Vec4b pixel = src.at<Vec4b>(i, j);
                dst.at<Vec4b>(i, j) = pixel;
            } else if(sizeof(T) == 4 && nc == 1){
                float pixel = src.at<float>(i, j);
                output_data[k++] = pixel;
            } else if(sizeof(T) == 4 && nc == 3){
                Vec3f pixel = src.at<Vec3f>(i, j);
                output_data[k++] = pixel[0];
                output_data[k++] = pixel[1];
                output_data[k++] = pixel[2];
            } else if(sizeof(T) == 4 && nc == 4){
                Vec4f pixel = src.at<Vec4f>(i, j);
                dst.at<Vec4f>(i, j) = pixel;
            }
        }
    }
    checkResult<T, nc>((T*)src.ptr(), output_data, output_height,
            output_width, step/sizeof(T), dstep/sizeof(T));
    return 0;

}



TEST_F(pixeldereferTest, none){
    imageParam p ;
 //   ASSERT_TRUE(!imageparams.empty());
   // for(int i =0; i<imageparams.size(); i++)
  //  {
        p = imagep;
       testPixelderefer(p);
   // }
       
}


