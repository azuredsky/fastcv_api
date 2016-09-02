#include "test_head.hpp"
using namespace HPC::fastcv;

struct imageParam{
    size_t width;
    size_t height;
    float space_sigma;
    float range_sigma;
};



class fastBilateralFilterTest : public testing::Test
{
      public:
        virtual void SetUp()
        {
            imageparams.clear();
            imageparams.swap(imageparams);
            std::vector<imageParam>::iterator iter = imageparams.begin();
            if(globalEnvironment::Instance()->testconfig.hasSection("fastBilateralFilterTest"))
            {
                std::cout<<"have"<<std::endl;
                Cfg::cfg_type vec_arg = globalEnvironment::Instance()->testconfig.sectionConfigVec("fastBilateralFilterTest");
                int num = vec_arg.size();
                for(int i = 0;i < num; i++)
                {
                    ASSERT_TRUE(vec_arg[i].find("height")!=vec_arg[i].end())<<"argument:height not found";
                    ASSERT_TRUE(vec_arg[i].find("width")!=vec_arg[i].end())<<"argument:width not found";
                 
                    ASSERT_TRUE(vec_arg[i].find("space_sigma")!=vec_arg[i].end())<<"argument:space_sigma not found";
                    ASSERT_TRUE(vec_arg[i].find("range_sigma")!=vec_arg[i].end())<<"argument:range_sigma not found";
                    imagep.height = std::stoul(vec_arg[i]["height"]);
                    imagep.width = std::stoul(vec_arg[i]["width"]);
                    imagep.range_sigma = std::stof(vec_arg[i]["range_sigma"]);
                    imagep.space_sigma = std::stof(vec_arg[i]["space_sigma"]);              
                    std::cout<<"size"<<imageparams.size()<<std::endl;                 
                    iter = imageparams.insert(iter,imagep);
                    std::cout<<"size"<<imageparams.size()<<std::endl;

                }
            }
            else
            {
                std::cout<<"dont have"<<std::endl;
               // imagep={64, 64, 0.5, 0.5};
                imagep.height = 64;
                imagep.width = 64;
                imagep.range_sigma = 0.5;
                imagep.space_sigma = 0.5;
                std::cout<<"size"<<imageparams.size()<<std::endl;
                imageparams.insert(iter,imagep);
                std::cout<<"size"<<imageparams.size()<<std::endl;
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


template<typename T>
void randomRangeData(T *data, const int max, const size_t num){
    size_t tmp;
    clock_t ct = clock();
    srand((unsigned int)ct);

    for(size_t i = 0; i < num; i++){
        tmp = rand()% max;
        data[i++] = (T)tmp;
    }
}

template<typename T, int ncSrc, int ncBase, int ncDst, int nc, BorderType bt>
int testFastBilateralFilterFunc(imageParam p) {
    size_t input_width = p.width;
    size_t input_height = p.height;
    size_t input_channels = nc;
    double space_sigma = p.space_sigma;
    double range_sigma = p.range_sigma;

    printf("Input Mat scale: width*height*channels = %ld*%ld*%ld\n", input_width, input_height, input_channels);

    size_t input_data_num = input_width*input_height*input_channels;

    T* input_data = (T*)malloc(input_data_num*sizeof(T));
    randomRangeData(input_data, 255, input_data_num);
    //convert to (0,1)
    for(size_t i=0; i<input_data_num; i++){
        input_data[i] = (T)(input_data[i]/255);
    }

    T* base_data = (T*)malloc(input_data_num*sizeof(T));
    randomRangeData(base_data, 255, input_data_num);
    //convert to (0,1)
    for(size_t i=0; i<input_data_num; i++){
        base_data[i] = (T)(base_data[i]/255);
    }

    Mat<T, ncSrc> src(input_height, input_width);
    src.fromHost(input_data);
    Mat<T, ncBase> base(input_height, input_width);
    base.fromHost(base_data);
    Mat<T, ncDst> dst(input_height, input_width);

    HPCStatus_t sta = fastBilateralFilter<bt, T, ncSrc, ncBase, ncDst, nc>(src, base, range_sigma, space_sigma, &dst);
    EXPECT_EQ(HPC_SUCCESS, sta);

    free(input_data);
    return 0;
}


TEST_F(fastBilateralFilterTest, Ffloat){
    imageParam p ;
  //  ASSERT_TRUE(!imageparams.empty());
 //   for(int i = 0;i < imageparams.size();i++)
  //  {
        p = imagep;
        testFastBilateralFilterFunc<float, 1, 1, 1, 1, BORDER_DEFAULT>(p);
        testFastBilateralFilterFunc<float, 3, 3, 3, 3, BORDER_DEFAULT>(p);
        testFastBilateralFilterFunc<float, 4, 4, 4, 4, BORDER_DEFAULT>(p);
 //   }
    
}
