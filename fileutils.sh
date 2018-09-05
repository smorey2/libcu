#! /bin/sh
cp src/CMakeLists.fileutils.txt src/CMakeLists.txt

# START
rm -rf _fileutils
mkdir _fileutils
cd _fileutils


# BUILD64
cmake -DM=64 -DC64=ON -DCMAKE_CUDA_FLAGS="-arch=sm_60" ../src
cmake --build . --config Debug


# END
cd ..