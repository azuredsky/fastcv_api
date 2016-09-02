mkdir benchmark_cuda
./bin/benchmark_cvtcolor 2>benchmark_cuda/cvtcolor_cuda.txt
./bin/benchmark_bilateral 2>benchmark_cuda/bilateral_cuda.txt
./bin/benchmark_boxfilter 2>benchmark_cuda/boxfilter_cuda.txt
./bin/benchmark_filter2d 2>benchmark_cuda/filter2d_cuda.txt
./bin/benchmark_gaussblur 2>benchmark_cuda/gaussblur_cuda.txt
./bin/benchmark_warpaffine 2>benchmark_cuda/warpaffine_cuda.txt
./bin/benchmark_resize 2>benchmark_cuda/resize_cuda.txt
./bin/benchmark_convertto 2>benchmark_cuda/convertto_cuda.txt
./bin/benchmark_crop 2>benchmark_cuda/crop_cuda.txt
./bin/benchmark_integral 2>benchmark_cuda/integral_cuda.txt