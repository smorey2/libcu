cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(libcu LANGUAGES CXX CUDA)

include_directories(include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
link_directories(../_build/build64)

# dcat
add_executable(dcat
	libcu.fileutils/dcat.cu
  )
  target_link_libraries(dcat PRIVATE libcu.${arch})
# dcd
add_executable(dcd
	libcu.fileutils/dcd.cu
  )
  target_link_libraries(dcd PRIVATE libcu.${arch})
# dchgrp
add_executable(dchgrp
	libcu.fileutils/dchgrp.cu
  )
  target_link_libraries(dchgrp PRIVATE libcu.${arch})
# dchmod
add_executable(dchmod
	libcu.fileutils/dchmod.cu
  )
  target_link_libraries(dchmod PRIVATE libcu.${arch})
# dchown
add_executable(dchown
	libcu.fileutils/dchown.cu
  )
  target_link_libraries(dchown PRIVATE libcu.${arch})
# dcmp
add_executable(dcmp
	libcu.fileutils/dcmp.cu
  )
  target_link_libraries(dcmp PRIVATE libcu.${arch})
# dcp
add_executable(dcp
	libcu.fileutils/dcp.cu
  )
  target_link_libraries(dcp PRIVATE libcu.${arch})
# dgrep
add_executable(dgrep
	libcu.fileutils/dgrep.cu
  )
  target_link_libraries(dgrep PRIVATE libcu.${arch})
# dls
add_executable(dls
	libcu.fileutils/dls.cu
  )
  target_link_libraries(dls PRIVATE libcu.${arch})
# dmkdir
add_executable(dmkdir
	libcu.fileutils/dmkdir.cu
  )
  target_link_libraries(dmkdir PRIVATE libcu.${arch})
# dmore
add_executable(dmore
	libcu.fileutils/dmore.cu
  )
  target_link_libraries(dmore PRIVATE libcu.${arch})
# dmv
add_executable(dmv
	libcu.fileutils/dmv.cu
  )
  target_link_libraries(dmv PRIVATE libcu.${arch})
# dpwd
add_executable(dpwd
	libcu.fileutils/dpwd.cu
  )
  target_link_libraries(dpwd PRIVATE libcu.${arch})
# drm
add_executable(drm
	libcu.fileutils/drm.cu
  )
  target_link_libraries(drm PRIVATE libcu.${arch})
# drmdir
add_executable(drmdir
	libcu.fileutils/drmdir.cu
  )
  target_link_libraries(drmdir PRIVATE libcu.${arch})

# Request that libcu be built with -std=c++11. As this is a public compile feature anything that links to particles will also build with -std=c++11
target_compile_features(dcat PUBLIC cxx_std_11)
target_compile_features(dcd PUBLIC cxx_std_11)
target_compile_features(dchgrp PUBLIC cxx_std_11)
target_compile_features(dchmod PUBLIC cxx_std_11)
target_compile_features(dchown PUBLIC cxx_std_11)
target_compile_features(dcmp PUBLIC cxx_std_11)
target_compile_features(dcp PUBLIC cxx_std_11)
target_compile_features(dgrep PUBLIC cxx_std_11)
target_compile_features(dls PUBLIC cxx_std_11)
target_compile_features(dmkdir PUBLIC cxx_std_11)
target_compile_features(dmore PUBLIC cxx_std_11)
target_compile_features(dmv PUBLIC cxx_std_11)
target_compile_features(dpwd PUBLIC cxx_std_11)
target_compile_features(drm PUBLIC cxx_std_11)
target_compile_features(drmdir PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle library to be built with -dc as the member functions could be called by other libraries and executables
set_target_properties(dcat PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(dcd PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(dchgrp PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(dchmod PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(dchown PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(dcmp PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(dcp PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(dgrep PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(dls PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(dmkdir PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(dmore PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(dmv PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(dpwd PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(drm PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(drmdir PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
if (M)
  set_target_properties(dcat PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
  set_target_properties(dcd PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
  set_target_properties(dchgrp PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
  set_target_properties(dchmod PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
  set_target_properties(dchown PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
  set_target_properties(dcmp PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
  set_target_properties(dcp PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
  set_target_properties(dgrep PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
  set_target_properties(dls PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
  set_target_properties(dmkdir PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
  set_target_properties(dmore PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
  set_target_properties(dmv PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
  set_target_properties(dpwd PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
  set_target_properties(drm PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
  set_target_properties(drmdir PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
endif()