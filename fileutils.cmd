@echo off
copy /Y src\CMakeLists.fileutils.txt src\CMakeLists.txt > nul

rem START
rd /s /q _fileutils
rem rm -rf _fileutils
mkdir _fileutils
pushd _fileutils


rem BUILD64
cmake -G "Visual Studio 15 2017 Win64" -DCMAKE_CUDA_FLAGS="-arch=sm_60" ../src
cmake --build . --config Debug


rem END
popd