#pragma once
#include <iostream>
#include "alsutils/error/Exception.hpp"
//! Executes the given code and checks for cuda error
#define CUDA_SAFE_CALL(x) { \
    cudaError_t error = x; \
    if (error != cudaSuccess) { \
        std::cerr << "Noticed CUDA error in " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::cerr << "\tLine was:\"" << #x << "\"" << std::endl; \
        std::cerr << "\tError: " << cudaGetErrorString(error) << std::endl; \
        THROW("CUDA error" << std::endl << "Line was: " << std::endl <<"\t" << #x << "\nError code: " << error); \
    } \
}

//! Does the same as CUDA_SAFE_CALL, but doesn't print an error message
#define CUDA_SAFE_CALL_SILENT(x) { \
    cudaError_t error = x; \
    if (error != cudaSuccess) { \
        THROW("CUDA error" << std::endl << "Line was: " << std::endl <<"\t" << #x << "\nError code: " << error); \
    } \
}
