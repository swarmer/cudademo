cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

include(CTest)

# library target
add_library(libcudademo STATIC
  src/cudademo.cu
  src/cudademo.h
)

# Request that libcudademo be built with -std=c++11
# As this is a public compile feature anything that links to
# libcudademo will also build with -std=c++11
target_compile_features(libcudademo PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the
# cudademo library to be built with -dc as the member functions
# could be called by other libraries and executables
set_target_properties(
    libcudademo
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)

# executable target
add_executable(cudademo_test src/cudademo_test.cu)
set_property(
    TARGET cudademo_test
    PROPERTY CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(cudademo_test PRIVATE libcudademo)

if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath,
  # so that the static cuda runtime can find it at runtime.
  set_property(
    TARGET cudademo_test
    PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
  )
endif()
