#include "ppl/ppl.h"
#include "ppl_help.h"
#include "../../ppl_test.h"

struct DataParam{
    size_t height;
    size_t width;
};

class IntegralImageTest: public testing::TestWithParam< DataParam>{};
struct DataParam datap;
INSTANTIATE_TEST_CASE_P(TrueReturn,  IntegralImageTest, testing::Values(datap));

int main(int argc, char *argv[]){
    if(argc !=1 && argc !=3){
        printf("Error Params!\n");
        printf("Usage: ./run height width\n");
        return -1;
    }
    if(argc ==3){
        datap.width = atoi(argv[1]);
        datap.height = atof(argv[2]);
    }
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
template<typename T>
int testIntegralImage(DataParam datap) {

    printf("IntegralImage: [image width*height]:[%ld, %ld]\n", datap.width, datap.height);

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
    ParrotsPrimitiveType dt = getParrotsPrimitiveType<T>();
    EXPECT_EQ_SUCCESS(parrotsSetContiguous2DArraySpec(inputDesc,
                dt, width, height));

    //create and set output tensor
    ParrotsDataSpec* outputDesc = NULL;
    EXPECT_EQ_SUCCESS(parrotsCreateDataSpec(&outputDesc));
    EXPECT_EQ_SUCCESS(parrotsSetContiguous2DArraySpec(outputDesc,
                dt, width, height));

    //X86 changeDataTypeAndScale
    T *input_data = (T*)malloc(data_size*sizeof(T));
    T *output_data = (T*)malloc(data_size*sizeof(T));

    randomRangedData(input_data, (T)255, data_size);

#if defined (PPL_USE_X86)
    EXPECT_EQ_SUCCESS(pplX86IntegralImage(handle_x86, inputDesc,
                (const void*)input_data, outputDesc, (void*)output_data));
#endif
#if defined (PPL_USE_ARM)
    EXPECT_EQ_SUCCESS(pplARMIntegralImage(handle_arm, inputDesc,
                (const void*)input_data, outputDesc, (void*)output_data));
#endif

    //cuda
    T *output_data_cuda = (T*)malloc(data_size*sizeof(T));

    //copy data to device
    T *d_input_data;
    T *d_output_data;
    cudaMalloc((void **)&d_input_data, data_size*sizeof(T));
    cudaMalloc((void **)&d_output_data, data_size*sizeof(T));
    cudaMemcpy(d_input_data, input_data, data_size*sizeof(T),
            cudaMemcpyHostToDevice);
    cudaMemset((void*)d_output_data, 0, data_size*sizeof(T));

    EXPECT_EQ_SUCCESS( pplCUDAIntegralImage(handle_cuda, inputDesc,
                (const void*)d_input_data, outputDesc, (void*)d_output_data));

    //copy data back to cpu
    cudaMemcpy(output_data_cuda, d_output_data, data_size*sizeof(T),
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

TEST_P(IntegralImageTest, IntegralI_CPU_GPU_UINT32){
    DataParam datap = GetParam();
    testIntegralImage<uint>(datap);
}

TEST_P(IntegralImageTest, IntegralI_CPU_GPU_FLOAT32) {
    DataParam datap = GetParam();
    testIntegralImage<float>(datap);
}

TEST_P(IntegralImageTest, IntegralI_CPU_GPU_FLOAT64) {
    DataParam datap = GetParam();
    testIntegralImage<double>(datap);
}
