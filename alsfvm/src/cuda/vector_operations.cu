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

	template<class T>
	__global__ void addKernel(T* result, const T* a, T b, size_t size) {
		size_t index = blockIdx.x*blockDim.x + threadIdx.x;

		if (index < size) {
			result[index] = a[index] + b;
		}
	}

	template<class T>
	__global__ void multiplyKernel(T* result, const T* a, T b, size_t size) {
		size_t index = blockIdx.x*blockDim.x + threadIdx.x;

		if (index < size) {
			result[index] = a[index] * b;
		}
	}

	template<class T>
	__global__ void divideKernel(T* result, const T* a, T b, size_t size) {
		size_t index = blockIdx.x*blockDim.x + threadIdx.x;

		if (index < size) {
			result[index] = a[index] / b;
		}
	}

	template<class T>
	__global__ void subtractKernel(T* result, const T* a, T b, size_t size) {
		size_t index = blockIdx.x*blockDim.x + threadIdx.x;

		if (index < size) {
			result[index] = a[index] - b;
		}
	}

    template<class T>
    __global__ void linear_combination_device(T a1, T* v1,
        T a2, const T* v2,
        T a3, const T* v3,
        T a4, const T* v4,
        T a5, const T* v5,
        size_t size) {
        size_t index = blockIdx.x*blockDim.x + threadIdx.x;

        if (index < size) {
            v1[index] = a1*v1[index] + a2*v2[index] + a3*v3[index] + a4*v4[index] + a5*v5[index];
        }
    }

}

// Since we use templates, we must instatiate.
#define INSTANTIATE_VECTOR_OPERATION(x)\
	template void x<real>(real*, const real*, const real*, size_t);

#define INSTANTIATE_VECTOR_SCALAR_OPERATION(x)\
	template void x<real>(real*, const real*, const real, size_t);



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
			addKernel << <(size + threadCount - 1) / threadCount, threadCount >> >(result, a, b, size);
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
			multiplyKernel << <(size + threadCount - 1) / threadCount, threadCount >> >(result, a, b, size);
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
			subtractKernel << <(size + threadCount - 1) / threadCount, threadCount >> >(result, a, b, size);
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
			divideKernel << <(size + threadCount - 1) / threadCount, threadCount >> >(result, a, b, size);
		}

		///
		/// Adds scalar to each component of b and stores the result to result
		/// \param result the device memory to write to (can be the same
		///               as a ). Must have length size (in T)
		/// \param a must have length size (in T)
		/// \param scalar the scalar
		///
		template<class T>
		void add(T* result, const T* a, T scalar, size_t size) {
			const size_t threadCount = 1024;
			addKernel << <(size + threadCount - 1) / threadCount, threadCount >> >(result, a, scalar, size);
		}

		///
		/// Multiplies scalar to each component of b and stores the result to result
		/// \param result the device memory to write to (can be the same
		///               as a ). Must have length size (in T)
		/// \param a must have length size (in T)
		/// \param scalar the scalar
		///
		template<class T>
		void multiply(T* result, const T* a, T scalar, size_t size) {
			const size_t threadCount = 1024;
			multiplyKernel << <(size + threadCount - 1) / threadCount, threadCount >> >(result, a, scalar, size);
		}

		///
		/// Subtracts scalar from each component of b and stores the result to result
		/// \param result the device memory to write to (can be the same
		///               as a ). Must have length size (in T)
		/// \param a must have length size (in T)
		/// \param scalar the scalar
		///
		template<class T>
		void subtract(T* result, const T* a, T scalar, size_t size) {
			const size_t threadCount = 1024;
			subtractKernel << <(size + threadCount - 1) / threadCount, threadCount >> >(result, a, scalar, size);
		}

		///
		/// Divides scalar from each component of b and stores the result to result
		/// \param result the device memory to write to (can be the same
		///               as a ). Must have length size (in T)
		/// \param a must have length size (in T)
		/// \param scalar the scalar
		///
		template<class T>
		void divide(T* result, const T* a, T scalar, size_t size) {
			const size_t threadCount = 1024;
			divideKernel <<<(size + threadCount - 1)/threadCount, threadCount >>>(result, a, scalar, size);
		}

        template<class T>
        void add_linear_combination(T a1, T* v1,
            T a2, const T* v2,
            T a3, const T* v3,
            T a4, const T* v4,
            T a5, const T* v5,
            size_t size) {
            const size_t threadCount = 1024;
            linear_combination_device << <(size + threadCount - 1) / threadCount, threadCount >> >(a1, v1, a2, v2, a3,v3, a4,v4,a5,v5, size);
        }

        INSTANTIATE_VECTOR_OPERATION(add)
            INSTANTIATE_VECTOR_OPERATION(subtract)
            INSTANTIATE_VECTOR_OPERATION(multiply)
            INSTANTIATE_VECTOR_OPERATION(divide)

            INSTANTIATE_VECTOR_SCALAR_OPERATION(add)
            INSTANTIATE_VECTOR_SCALAR_OPERATION(subtract)
            INSTANTIATE_VECTOR_SCALAR_OPERATION(multiply)
            INSTANTIATE_VECTOR_SCALAR_OPERATION(divide)

            template void add_linear_combination<real>(real a1, real* v1, real a2, const real* v2,
                real a3, const real* v3,
                real a4, const real* v4,
                real a5, const real* v5,
                size_t size);

	}
}
