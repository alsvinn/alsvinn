#include "alsfvm/cuda/vector_operations.hpp"
#include "alsfvm/types.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

namespace {
	template<class T>
	__global__ void addKernel(T* result, const T* a, const T* b, size_t size) {
		size_t index = blockIdx.x*blockDim.x + threadIdx.x;

		if (index < size) {
			result[index] = a[index] + b[index];
		}
	}

	template<class T>
	__global__ void multiplyKernel(T* result, const T* a, const T* b, size_t size) {
		size_t index = blockIdx.x*blockDim.x + threadIdx.x;

		if (index < size) {
			result[index] = a[index] * b[index];
		}
	}

	template<class T>
	__global__ void divideKernel(T* result, const T* a, const T* b, size_t size) {
		size_t index = blockIdx.x*blockDim.x + threadIdx.x;

		if (index < size) {
			result[index] = a[index]/b[index];
		}
	}

	template<class T>
	__global__ void subtractKernel(T* result, const T* a, const T* b, size_t size) {
		size_t index = blockIdx.x*blockDim.x + threadIdx.x;

		if (index < size) {
			result[index] = a[index] - b[index];
		}
	}
}

// Since we use templates, we must instatiate.
#define INSTANTIATE_VECTOR_OPERATION(x)\
	template void x<real>(real*, const real*, const real*, size_t);


namespace alsfvm {
	namespace cuda {

		///
		/// Adds a and b and stores the result to result
		/// \param result the device memory to write to (can be the same
		///               as a or b). Must have length size (in T)
		/// \param a must have length size (in T)
		/// \param b must have length size (in T)
		///
		template<class T>
		void add(T* result, const T* a, const T* b, size_t size) {
			const size_t threadCount = 1024;
			addKernel <<<(size + threadCount - 1), threadCount >>>(result, a, b, size);
		}

		///
		/// Multiplies a and b and stores the result to result
		/// \param result the device memory to write to (can be the same
		///               as a or b). Must have length size (in T)
		/// \param a must have length size (in T)
		/// \param b must have length size (in T)
		///
		template<class T>
		void multiply(T* result, const T* a, const T* b, size_t size) {
			const size_t threadCount = 1024;
			multiplyKernel <<<(size + threadCount - 1), threadCount >>>(result, a, b, size);
		}

		///
		/// Subtracts a and b and stores the result to result
		/// \param result the device memory to write to (can be the same
		///               as a or b). Must have length size (in T)
		/// \param a must have length size (in T)
		/// \param b must have length size (in T)
		///
		template<class T>
		void subtract(T* result, const T* a, const T* b, size_t size) {
			const size_t threadCount = 1024;
			subtractKernel <<<(size + threadCount - 1), threadCount >>>(result, a, b, size);
		}
		///
		/// Divides a and b and stores the result to result
		/// \param result the device memory to write to (can be the same
		///               as a or b). Must have length size (in T)
		/// \param a must have length size (in T)
		/// \param b must have length size (in T)
		///
		template<class T>
		void divide(T* result, const T* a, const T* b, size_t size) {
			const size_t threadCount = 1024;
			divideKernel <<<(size + threadCount - 1), threadCount >>>(result, a, b, size);
		}

		INSTANTIATE_VECTOR_OPERATION(add)
		INSTANTIATE_VECTOR_OPERATION(subtract)
		INSTANTIATE_VECTOR_OPERATION(multiply)
		INSTANTIATE_VECTOR_OPERATION(divide)
	}
}
