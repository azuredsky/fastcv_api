set(BENCHMARK_LINK_LIBS
    api_fastcv
    opencv_imgproc
    opencv_highgui
    opencv_core
    ${CUDA_LIBRARIES}
    )
if(USE_CUDA)
    set(BENCHMARK_LINK_LIBS ${BENCHMARK_LINK_LIBS} opencv_gpu)
endif()
if(USE_OPENCL)
    set(BENCHMARK_LINK_LIBS ${BENCHMARK_LINK_LIBS} opencv_gpu opencv_ocl)
endif()

set(BENCHMARK_UNITS
    benchmark_cvtcolor
    benchmark_bilateral
    benchmark_boxfilter
    benchmark_filter2d
    benchmark_gaussblur
    benchmark_warpaffine
    benchmark_resize
    benchmark_convertto
    benchmark_crop
    benchmark_integral)

foreach(tname ${BENCHMARK_UNITS})
    add_executable(${tname} ${tname}.cpp)
    target_link_libraries(${tname} ${BENCHMARK_LINK_LIBS})
endforeach()

