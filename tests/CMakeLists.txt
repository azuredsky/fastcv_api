set(TEST_BASE
	base/test_resize
	base/test_filter
	base/test_gaussianBlur
    base/test_cvtColor
    base/test_convertTo
    base/test_crop
    base/test_warpAffine
    base/test_boxFilter
    base/test_integral
    base/test_fastBilateralFilter
    base/test_bilateralFilter
    base/test_readImage
    base/test_I420
    base/test_rotate
    base/test_rotate_yuv420
    base/test_pixel_derefer
    base/test_remap
    base/test_erode
    base/test_blur
    base/test_dilate
	base/test_encode_decode
)

if(USE_CUDA)
    SET(TEST_BASE
        ${TEST_BASE}
        base/test_half_crop
        base/test_half_convertTo
        base/test_half_cvtColor
        base/test_half_integral
        base/test_half_resize
        base/test_half_warpAffine
#base/test_half_boxFilter
#       base/test_half_gaussianBlur
#base/test_half_bilateralFilter
#       base/test_half_filter
        )
endif()
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR})
#foreach(tname ${TEST_BASE})
add_executable(test_all ${TEST_BASE} test_main.cpp cfg.cpp)
    target_link_libraries(test_all ${TESTS_LINK_LIBS})
    #add_test(NAME ${tname} COMMAND ${tname})
    if(CPPCHECK_FOUND)
        add_cppcheck_sources(${tname} base/${tname}.cpp)
    endif()
	#endforeach()
enable_testing()

if(BUILD_BENCHMARK)
    add_subdirectory(./benchmark)
endif()

# Install scripts directory
install(DIRECTORY ./scripts DESTINATION ${CMAKE_BINARY_DIR}
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ
    GROUP_EXECUTE WORLD_READ
    DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ
    GROUP_EXECUTE WORLD_READ)
