cmake_minimum_required(VERSION 3.18)

project(cuda_flood_fill LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

add_executable(flood_fill_cuda flood_fill_cuda.cu)

set_target_properties(flood_fill_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)