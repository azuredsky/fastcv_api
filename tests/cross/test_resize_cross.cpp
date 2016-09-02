#include "ppl/ppl.h"
#include "../../ppl_test.h"
#include "ppl_help.h"

struct ImageParam{
    size_t srcWidth;
    size_t srcHeight;
    size_t srcChannels;
    size_t destWidth;
    size_t destHeight;
    size_t destChannels;
    bool mode_linear;
};

class ResizeImageTest : public testing::TestWithParam<ImageParam>{};
struct ImageParam imgp;
INSTANTIATE_TEST_CASE_P(TrueReturn, ResizeImageTest, testing::Values(imgp));

int main(int argc, char *argv[]){
    if(argc !=1 && argc !=8){
        printf("Error Params!\n");
        printf("Usage: ./run src_image_width src_image_height src_channels dest_image_width dest_image_Height dest_channels mode\n");
        return -1;
    }
    imgp = {10, 10, 1, 15, 15, 1, 1};
    if(argc == 8){
        imgp.srcWidth = atoi(argv[1]);
        imgp.srcHeight = atoi(argv[2]);
        imgp.srcChannels = atoi(argv[3]);
        imgp.destWidth = atoi(argv[4]);
        imgp.destHeight = atoi(argv[5]);
        imgp.destChannels = atoi(argv[6]);
        imgp.mode_linear = atoi(argv[7]);
    }

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
template<typename T>
int testResizeImage(ImageParam imgp) {

    printf("Resize Image: [src_image witdh*height*channels]: [%ld, %ld, %ld]\n \
                 [dest_image witdh*height*channels]: [%ld, %ld, %ld]\n \
                 mode is linear: [%d]\n",
            imgp.srcWidth, imgp.srcHeight, imgp.srcChannels, imgp.destWidth, imgp.destHeight, imgp.destChannels, imgp.mode_linear);

    size_t input_width = imgp.srcWidth;
    size_t input_height = imgp.srcHeight;
    size_t input_channels = imgp.srcChannels;

    size_t output_width = imgp.destWidth;
    size_t output_height = imgp.destHeight;
    size_t output_channels = imgp.destChannels;

    bool mode_linear = imgp.mode_linear;


    size_t input_data_num = input_width*input_height*input_channels;
    size_t output_data_num = output_width*output_height*output_channels;

    //create handle
    size_t core = 2;
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
    EXPECT_EQ_SUCCESS(parrotsSetContiguous4DArraySpec(inputDesc,
                dt, input_width, input_height, input_channels, 1));

    //create and set output tensor
    ParrotsDataSpec* outputDesc = NULL;
    EXPECT_EQ_SUCCESS(parrotsCreateDataSpec(&outputDesc));
    EXPECT_EQ_SUCCESS(parrotsSetContiguous4DArraySpec(outputDesc,
                dt, output_width, output_height, output_channels, 1));

    //get input data
    T *input_data = (T*)malloc(input_data_num*sizeof(T));
    T *output_data = (T*)malloc(output_data_num*sizeof(T));

    randomRangedData(input_data, (T)255, input_data_num);

    //x86 arm Resize
#if defined (PPL_USE_X86)
    EXPECT_EQ_SUCCESS(pplX86Resize(handle_x86, inputDesc,
                (const void*)input_data, outputDesc, (void*)output_data, mode_linear));
#endif

#if defined (PPL_USE_ARM)
    EXPECT_EQ_SUCCESS(pplARMResize(handle_arm, inputDesc,
                (const void*)input_data, outputDesc, (void*)output_data, mode_linear));
#endif


    T *output_data_cuda = (T*)malloc(output_data_num*sizeof(T));

    //copy data to device
    T *d_input_data, *d_output_data;
    cudaMalloc((void **)&d_input_data, input_data_num*sizeof(T));
    cudaMalloc((void **)&d_output_data, output_data_num*sizeof(T));
    cudaMemcpy(d_input_data, input_data, input_data_num*sizeof(T),
            cudaMemcpyHostToDevice);
    cudaMemset((void*)d_output_data, 0, output_data_num*sizeof(T));

    EXPECT_EQ_SUCCESS(pplCUDAResize(handle_cuda, inputDesc,
                (const void*)d_input_data, outputDesc, (void*)d_output_data, !mode_linear));

    //copy data back to cpu
    cudaMemcpy(output_data_cuda, d_output_data, output_data_num*sizeof(T),
            cudaMemcpyDeviceToHost);

    //check result
    //checkResult(output_data, output_data_cuda, output_data_num);

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

TEST_P(ResizeImageTest, Resize_CPU_GPU_UINT8){
    ImageParam imgp = GetParam();
    testResizeImage<uchar>(imgp);
}

