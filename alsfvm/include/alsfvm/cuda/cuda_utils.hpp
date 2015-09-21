#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include <iostream>
#include <exception>
#include "alsfvm/error/Exception.hpp"

#define CUDA_SAFE_CALL(x) { \
	if (x != cudaSuccess) { \
		std::cerr << "Noticed CUDA error in " << __FILE__ << ":" << __LINE__ << std::endl; \
		std::cerr << "\tLine was:\"" << #x << "\"" << std::endl; \
		THROW("CUDA error"); \
	} \
}

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