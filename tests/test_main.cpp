/*************************************************************************
	> File Name: test_main.cpp
	> Author: ma6174
	> Mail: ma6174@163.com 
	> Created Time: 2016年08月09日 星期二 11时10分30秒
 ************************************************************************/

#include <test_head.hpp>
//extern Cfg testconfig;
/*
class globalEnvironment : public ::testing::Environment
{
    public:
        virtual void SetUp()
        {
            testconfig->setCfgFile("/home/liuping/testmodify/fastcv_api/tests/configure/testconfigure.txt");
            EXPECT_TRUE(testconfig->hasAnyConfig());

        }
        virtual void TearDown()
        {

        }
        static globalEnvironment* Instance()
        {
            if(!p_instance_)
                p_instance_ = new globalEnvironment;
            return p_instance_;
        }
        Cfg *testconfig;
    public:
        static globalEnvironment* p_instance_;
};*/

globalEnvironment *globalEnvironment::p_instance_ = nullptr;
int main(int argc,char *argv[])
{
//	testconfig.setCfgFile("./configure/testconfigure.txt");
//	if(!testconfig.hasAnyConfig())
//	{
//		printf("configure file is empty!\n");
//		return 0;
//	}
	
	printf("test begin!\n");
	::testing::InitGoogleTest(&argc,argv);
    
    ::testing::AddGlobalTestEnvironment(globalEnvironment::Instance());
	return RUN_ALL_TESTS();
	
}
