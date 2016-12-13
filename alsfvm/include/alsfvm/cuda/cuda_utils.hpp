#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include <iostream>
#include <exception>
#include "alsutils/error/Exception.hpp"

#define CUDA_SAFE_CALL(x) { \
	cudaError_t error = x; \
	if (error != cudaSuccess) { \
		std::cerr << "Noticed CUDA error in " << __FILE__ << ":" << __LINE__ << std::endl; \
		std::cerr << "\tLine was:\"" << #x << "\"" << std::endl; \
		std::cerr << "\tError: " << cudaGetErrorString(error) << std::endl; \
		THROW("CUDA error" << std::endl << "Line was: " << std::endl <<"\t" << #x << "\nError code: " << error); \
	} \
}

#ifdef NDEBUG
#define CUDA_CHECK_IF_DEBUG
#else 
//! If in debug mode, checks if there has been any error
//! with cuda. 
#define CUDA_CHECK_IF_DEBUG { \
    CUDA_SAFE_CALL(cudaGetLastError()); \
CUDA_SAFE_CALL(cudaDeviceSynchronize()); \
CUDA_SAFE_CALL(cudaGetLastError()); \
}
#endif
namespace alsfvm {
	namespace cuda {
		inline dim3 calculateBlockDimensions(size_t numberOfXCells, size_t numberOfYCells, size_t numberOfZCells) {
			const size_t blockSize = 1024;
			return dim3(blockSize, numberOfYCells > 1 ? blockSize : 1, numberOfZCells > 1 ? blockSize : 1);
		}

		inline dim3 calculateGridDimensions(size_t numberOfXCells, size_t numberOfYCells, size_t numberOfZCells, dim3 blockDimensions) {
			return dim3((numberOfXCells + blockDimensions.x - 1) / blockDimensions.x,
				(numberOfYCells + blockDimensions.y - 1) / blockDimensions.y,
				(numberOfZCells + blockDimensions.z - 1) / blockDimensions.z);
		}
	}
}