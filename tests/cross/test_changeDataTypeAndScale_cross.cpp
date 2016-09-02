#include <limits>
#include "ppl/ppl.h"
#include "ppl_help.h"
#include "../../ppl_test.h"

struct DataParam{
    float scale;
    size_t height;
    size_t width;
};

class changeDataTypeAndScaleTest: public testing::TestWithParam< DataParam>{};
struct DataParam datap;
INSTANTIATE_TEST_CASE_P(TrueReturn, changeDataTypeAndScaleTest, testing::Values(datap));

int main(int argc, char *argv[]){
    if(argc !=1 && argc !=4){
        printf("Error Params!\n");
        printf("Usage: ./run scale height width\n");
        return -1;
    }
    datap = {1.5, 10, 10};
    if(argc ==4){
        datap.scale = atof(argv[1]);
        datap.height = atoi(argv[2]);
        datap.width = atoi(argv[3]);
    }

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
template<typename TIN, typename TOUT>
int testChangeDataTypeAndScale(DataParam datap) {

    printf("changeDataTypeAndScale: [Data scale*heigth*width]:[%f, %ld, %ld]\n",
            datap.scale, datap.width, datap.height);

    float scale = datap.scale;
    size_t width = datap.width;
    size_t height = datap.height;

    size_t data_size = width*height;

    //create handle
    size_t core = 0;
#if defined (PPL_USE_X86)
    X86Handle_t handle_x86;
    EXPECT_EQ_SUCCESS(pplX86CreateHandle(&handle_x86, core));
#endif
#if defined (PPL_USE_ARM)
    X86Handle_t handle_arm;
    EXPECT_EQ_SUCCESS(pplARMCreateHandle(&handle_arm, core));
#endif

    CUDAHandle_t handle_cuda;
    EXPECT_EQ_SUCCESS(pplCUDACreateHandle(&handle_cuda, core, NULL));

    //create and set input tensor
    ParrotsDataSpec* inputDesc = NULL;
    EXPECT_EQ_SUCCESS(parrotsCreateDataSpec(&inputDesc));
    ParrotsPrimitiveType dt_in = getParrotsPrimitiveType<TIN>();
    EXPECT_EQ_SUCCESS(parrotsSetContiguous2DArraySpec(inputDesc,
                dt_in, width, height));

    //create and set output tensor
    ParrotsDataSpec* outputDesc = NULL;
    EXPECT_EQ_SUCCESS(parrotsCreateDataSpec(&outputDesc));
    ParrotsPrimitiveType dt_out = getParrotsPrimitiveType<TOUT>();
    EXPECT_EQ_SUCCESS(parrotsSetContiguous2DArraySpec(outputDesc,
                dt_out, width, height));

    //X86 changeDataTypeAndScale
    TIN *input_data = (TIN*)malloc(data_size*sizeof(TIN));
    TOUT *output_data = (TOUT*)malloc(data_size*sizeof(TOUT));

    TIN max = std::numeric_limits<TIN>::max();
    if(max>(TIN)1000)
        max = (TIN)1000;
    randomRangedData<TIN>(input_data, max, data_size);

#if defined (PPL_USE_X86)
    EXPECT_EQ_SUCCESS(pplX86changeDataTypeAndScale(handle_x86, inputDesc,
                (const void*)input_data, scale, outputDesc, (void*)output_data));
#endif
#if defined (PPL_USE_ARM)
    EXPECT_EQ_SUCCESS(pplARMchangeDataTypeAndScale(handle_arm, inputDesc,
                (const void*)input_data, scale, outputDesc, (void*)output_data));
#endif

    //cuda changeDataTypeAndScale
    TOUT *output_data_cuda = (TOUT*)malloc(data_size*sizeof(TOUT));

    //copy data to device
    TIN *d_input_data;
    TOUT *d_output_data;
    cudaMalloc((void **)&d_input_data, data_size*sizeof(TIN));
    cudaMalloc((void **)&d_output_data, data_size*sizeof(TOUT));
    cudaMemcpy(d_input_data, input_data, data_size*sizeof(TIN),
            cudaMemcpyHostToDevice);
    cudaMemset((void*)d_output_data, 0, data_size*sizeof(TOUT));

    EXPECT_EQ_SUCCESS(pplCUDAchangeDataTypeAndScale(handle_cuda, inputDesc,
                (const void*)d_input_data, scale, outputDesc, (void*)d_output_data));

    //copy data back to cpu
    cudaMemcpy(output_data_cuda, d_output_data, data_size*sizeof(TOUT),
            cudaMemcpyDeviceToHost);

    checkResult(output_data, output_data_cuda, data_size);

    //destroy and free data
    EXPECT_EQ_SUCCESS(parrotsDestroyDataSpec(inputDesc));
    EXPECT_EQ_SUCCESS(parrotsDestroyDataSpec(outputDesc));
#if defined (PPL_USE_X86)
    EXPECT_EQ_SUCCESS(pplX86DestroyHandle(handle_x86));
#endif
#if defined (PPL_USE_ARM)
    EXPECT_EQ_SUCCESS(pplARMDestroyHandle(handle_arm));
#endif
    EXPECT_EQ_SUCCESS(pplCUDADestroyHandle(handle_cuda));

    free(input_data);
    free(output_data);

    cudaFree(d_input_data);
    cudaFree(d_output_data);
    free(output_data_cuda);

    return 0;

}

TEST_P(changeDataTypeAndScaleTest, ChangeDTS_CPU_GPU_UINT8) {
    DataParam datap = GetParam();
    testChangeDataTypeAndScale<uchar, ushort>(datap);
    testChangeDataTypeAndScale<uchar, uint>(datap);
    testChangeDataTypeAndScale<uchar, float>(datap);
    testChangeDataTypeAndScale<uchar, double>(datap);
}

TEST_P(changeDataTypeAndScaleTest, ChangeDTS_CPU_GPU_UINT16) {
    DataParam datap = GetParam();
    testChangeDataTypeAndScale<ushort, uchar>(datap);
    testChangeDataTypeAndScale<ushort, uint>(datap);
    testChangeDataTypeAndScale<ushort, float>(datap);
    testChangeDataTypeAndScale<ushort, double>(datap);
}

TEST_P(changeDataTypeAndScaleTest, ChangeDTS_CPU_GPU_UINT32) {
    DataParam datap = GetParam();
    testChangeDataTypeAndScale<uint, uchar>(datap);
    testChangeDataTypeAndScale<uint, ushort>(datap);
    testChangeDataTypeAndScale<uint, float>(datap);
    testChangeDataTypeAndScale<uint, double>(datap);
}

TEST_P(changeDataTypeAndScaleTest, ChangeDTS_CPU_GPU_FLOAT) {
    DataParam datap = GetParam();
    testChangeDataTypeAndScale<float, uchar>(datap);
    testChangeDataTypeAndScale<float, ushort>(datap);
    testChangeDataTypeAndScale<float, uint>(datap);
    testChangeDataTypeAndScale<float, double>(datap);
}

TEST_P(changeDataTypeAndScaleTest, ChangeDTS_CPU_GPU_DOUBLE) {
    DataParam datap = GetParam();
    testChangeDataTypeAndScale<double, uchar>(datap);
    testChangeDataTypeAndScale<double, ushort>(datap);
    testChangeDataTypeAndScale<double, uint>(datap);
    testChangeDataTypeAndScale<double, float>(datap);
}
