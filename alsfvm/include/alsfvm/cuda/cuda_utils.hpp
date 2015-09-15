#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include <iostream>
#include <exception>

#define CUDA_SAFE_CALL(x) { \
	if (x != cudaSuccess) { \
		std::cerr << "Noticed CUDA error in " << __FILE__ << ":" << __LINE__ << std::endl; \
		std::cerr << "\tLine was:\"" << #x << "\"" << std::endl; \
		throw std::runtime_error("CUDA error"); \
	} \
}
