/*************************************************************************
	> File Name: test_head.hpp
	> Author: ma6174
	> Mail: ma6174@163.com 
	> Created Time: 2016年08月09日 星期二 12时02分33秒
 ************************************************************************/

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <fastcv.hpp>
#include <cfg.h>
using namespace HPC::fastcv;
using namespace std;

class globalEnvironment : public ::testing::Environment
{
    public:
        globalEnvironment(){}
        virtual void SetUp()
        {
            printf("setup start\n");
            testconfig.setCfgFile("../../tests/configure/testconfigure.txt");
            EXPECT_TRUE(testconfig.hasAnyConfig());
            printf("setup end\n");
        }
        virtual void TearDown()
        {
            //testconfig = nullptr;

        }
        static globalEnvironment* Instance()
        {
            if(!p_instance_)
                p_instance_ = new globalEnvironment;
            return p_instance_;
        }
    public:
        Cfg testconfig;
        static globalEnvironment* p_instance_;
};


inline double timingExec(struct timeval start, struct timeval end)
{
	double timeuse = 1000.0*(end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000.0;
	return timeuse;
}






