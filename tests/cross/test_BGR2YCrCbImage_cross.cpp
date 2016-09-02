#include "ppl/ppl.h"
#include "ppl_help.h"
#include "../../ppl_test.h"

struct ImageParam{
    size_t width;
    size_t height;
    size_t channels;
};

class BGR2YCrCbImageTest : public testing::TestWithParam<ImageParam>{};
struct ImageParam imgp;
INSTANTIATE_TEST_CASE_P(TrueReturn, BGR2YCrCbImageTest, testing::Values(imgp));

int main(int argc, char *argv[]){
    if(argc !=1 && argc !=4){
        printf("Error Params!\n");
        printf("Usage: ./run image_width image_height channels\n");
        return -1;
    }
    imgp = {100, 100, 3};
    if(argc ==4){
        imgp.width = atoi(argv[1]);
        imgp.height = atoi(argv[2]);
        imgp.channels = atoi(argv[3]);
    }

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
template<typename T>
int testBGR2YCrCbImage(ImageParam imgp) {

    printf("BGR2YCrCbImage: [image witdh*height*channels]: [%ld, %ld, %ld]\n",
            imgp.width, imgp.height, imgp.channels);

    size_t image_width = imgp.width;
    size_t image_height = imgp.height;
    size_t channels = imgp.channels;

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
                dt, image_width, image_height));

    //create and set output tensor
    ParrotsDataSpec* outputDesc = NULL;
    EXPECT_EQ_SUCCESS(parrotsCreateDataSpec(&outputDesc));
    EXPECT_EQ_SUCCESS(parrotsSetContiguous2DArraySpec(outputDesc,
                dt, image_width, image_height));

    //get input data
    T *input_data = (T*)malloc(data_num*sizeof(T));
    T *output_data = (T*)malloc(data_num*sizeof(T));

    randomRangedData(input_data, (T)255, data_num);


    //x86 arm BGR2YCrCb
#if defined (PPL_USE_X86)
    EXPECT_EQ_SUCCESS(pplX86BGR2YCrCbImage(handle_x86, inputDesc,
                (const void*)input_data, outputDesc, (void*)output_data));
#endif

#if defined (PPL_USE_ARM)
    EXPECT_EQ_SUCCESS(pplARMBGR2YCrCbImage(handle_arm, inputDesc,
                (const void*)input_data, outputDesc, (void*)output_data));
#endif

    //cuda BGR2YCrCb
    T *output_data_cuda = (T*)malloc(data_num*sizeof(T));

    //copy data to device
    T *d_input_data, *d_output_data;
    cudaMalloc((void **)&d_input_data, data_num*sizeof(T));
    cudaMalloc((void **)&d_output_data, data_num*sizeof(T));
    cudaMemcpy(d_input_data, input_data, data_num*sizeof(T),
            cudaMemcpyHostToDevice);
    cudaMemset((void*)d_output_data, 0, data_num*sizeof(T));

    EXPECT_EQ_SUCCESS(pplCUDABGR2YCrCbImage(handle_cuda, inputDesc,
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
    free(input_data);
    free(output_data);

    EXPECT_EQ_SUCCESS(pplCUDADestroyHandle(handle_cuda));
    cudaFree(d_input_data);
    cudaFree(d_output_data);
    free(output_data_cuda);

    return 0;
}

TEST_P(BGR2YCrCbImageTest, BGRYCrCb_CPU_GPU_UINT8){
    ImageParam imgp = GetParam();
    testBGR2YCrCbImage<uchar>(imgp);
}

TEST_P(BGR2YCrCbImageTest, BGRYCrCb_CPU_GPU_UINT16){
    ImageParam imgp = GetParam();
    testBGR2YCrCbImage<ushort>(imgp);
}

TEST_P(BGR2YCrCbImageTest, BGRYCrCb_CPU_GPU_UINT32){
    ImageParam imgp = GetParam();
    testBGR2YCrCbImage<uint>(imgp);
}
