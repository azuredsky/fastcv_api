#include "ppl/ppl.h"
#include "ppl_help.h"
#include "../../ppl_test.h"

struct ImageParam{
    size_t width;
    size_t height;
    size_t channels;
};

class YCrCb2BGRImageTest : public testing::TestWithParam<ImageParam>{};
struct ImageParam imagep;
INSTANTIATE_TEST_CASE_P(TrueReturn, YCrCb2BGRImageTest, testing::Values(imagep));

int main(int argc, char *argv[]){
    if(argc !=1 && argc !=4){
        printf("Error Params!\n");
        printf("Usage: ./run image_width image_height channels\n");
        return -1;
    }
    if(argc==4){
        imagep.width = atoi(argv[1]);
        imagep.height = atoi(argv[2]);
        imagep.channels = atoi(argv[3]);
    }
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
template<typename T>
int testYCrCb2BGRImage(ImageParam imgp) {

    printf("YCrCb2BGRImage:[image width*height*channels]:[ %ld, %ld, %ld]",
            imgp.width, imgp.height, imgp.channels);

    size_t image_width = imgp.width;
    size_t image_height = imgp.height;
    size_t channels = imgp.channels;

    size_t input_width = channels;
    size_t input_height = image_width*image_height;

    size_t output_width = input_width;
    size_t output_height = input_height;

    size_t data_num = image_width*image_height*channels;

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
                dt, input_width, input_height));

    //create and set output tensor
    ParrotsDataSpec* outputDesc = NULL;
    EXPECT_EQ_SUCCESS(parrotsCreateDataSpec(&outputDesc));
    EXPECT_EQ_SUCCESS(parrotsSetContiguous2DArraySpec(outputDesc,
                dt, output_width, output_height));

    //get input data
    T *input_data = (T*)malloc(data_num*sizeof(T));
    T *output_data = (T*)malloc(data_num*sizeof(T));

    randomRangedData(input_data, (T)100, data_num);

    //x86 arm YCrCb2BGR
#if defined (PPL_USE_X86)
    EXPECT_EQ_SUCCESS(pplX86YCrCb2BGRImage(handle_x86, inputDesc,
                (const void*)input_data, outputDesc, (void*)output_data));
#endif

#if defined (PPL_USE_ARM)
    EXPECT_EQ_SUCCESS(pplARMYCrCb2BGRImage(handle_arm, inputDesc,
                (const void*)input_data, outputDesc, (void*)output_data));
#endif

    //cuda YCrCb2BGR
    T *output_data_cuda = (T*)malloc(data_num*sizeof(T));

    //copy data to device
    T *d_input_data, *d_output_data;
    cudaMalloc((void **)&d_input_data, data_num*sizeof(T));
    cudaMalloc((void **)&d_output_data, data_num*sizeof(T));
    cudaMemcpy(d_input_data, input_data, data_num*sizeof(T),
            cudaMemcpyHostToDevice);
    cudaMemset((void*)d_output_data, 0, data_num*sizeof(T));

    EXPECT_EQ_SUCCESS(pplCUDAYCrCb2BGRImage(handle_cuda, inputDesc,
                (const void*)d_input_data, outputDesc, (void*)d_output_data));

    //copy data back to cpu
    cudaMemcpy(output_data_cuda, d_output_data, data_num*sizeof(T),
            cudaMemcpyDeviceToHost);

    //check result
    checkResult(output_data, output_data_cuda, data_num);

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

TEST_P(YCrCb2BGRImageTest, YCrCbBGR_CPU_GPU_UINT8) {
    ImageParam imgp = GetParam();
    testYCrCb2BGRImage<uchar>(imgp);
}

TEST_P(YCrCb2BGRImageTest, YCrCbBGR_CPU_GPU_UINT16) {
    ImageParam imgp = GetParam();
    testYCrCb2BGRImage<ushort>(imgp);
}

TEST_P(YCrCb2BGRImageTest, YCrCbBGR_CPU_GPU_UINT32) {
    ImageParam imgp = GetParam();
    testYCrCb2BGRImage<uint>(imgp);
}

TEST_P(YCrCb2BGRImageTest, YCrCbBGR_CPU_GPU_FLOAT32) {
    ImageParam imgp = GetParam();
    testYCrCb2BGRImage<float>(imgp);
}

TEST_P(YCrCb2BGRImageTest, YCrCbBGR_CPU_GPU_FLOAT64) {
    ImageParam imgp = GetParam();
    testYCrCb2BGRImage<double>(imgp);
}
