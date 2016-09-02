./bin/test_convertTo
./bin/test_convertTo 48 48 48 48
./bin/test_convertTo 24 24 24 24
./bin/test_convertTo 256 256 256 256

./bin/test_crop
./bin/test_crop 64 64 28 40
./bin/test_crop 128 256 64 128
./bin/test_crop 120 140 100 48

./bin/test_cvtColor
./bin/test_cvtColor 64 80 64 80
./bin/test_cvtColor 256 128 256 128
./bin/test_cvtColor 1000 888 1000 888

./bin/test_resize 200 190 300 210
./bin/test_resize 87 34 76 490
./bin/test_resize 128 200 200 128
./bin/test_resize 1000 5000 2000 3000

./bin/test_warpAffine
./bin/test_warpAffine 64 88 64 88
./bin/test_warpAffine 120 240 120 240
./bin/test_warpAffine 1000 2000 1000 200

