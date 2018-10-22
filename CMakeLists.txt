cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cudademo LANGUAGES CXX CUDA)

include(CTest)


# library target
################
add_library(nbodycuda STATIC
  src/nbodycuda.cu
  src/nbodycuda.h
)

# Request that nbodycuda be built with -std=c++11
# As this is a public compile feature anything that links to
# nbodycuda will also build with -std=c++11
target_compile_features(nbodycuda PUBLIC cxx_std_14)

# We need to explicitly state that we need all CUDA files in the
# nbodycuda library to be built with -dc as the member functions
# could be called by other libraries and executables
set_target_properties(
    nbodycuda
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)


# executable target
###################
find_package(SDL2 REQUIRED)
include_directories(nbody ${SDL2_INCLUDE_DIRS})

add_executable(nbody src/nbody.cpp)
set_property(
    TARGET nbody
    PROPERTY CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(nbody PRIVATE nbodycuda ${SDL2_LIBRARIES})

if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath,
  # so that the static cuda runtime can find it at runtime.
  set_property(
    TARGET nbody
    PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
  )
endif()
