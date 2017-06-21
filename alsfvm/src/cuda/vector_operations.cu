#include "alsfvm/cuda/vector_operations.hpp"
#include "alsfvm/types.hpp"
#include "alsfvm/cuda/cuda_utils.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

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


    template<class T>
    __global__ void add_power_device(T* out, const T* a, double power, size_t size) {
        size_t index = blockIdx.x*blockDim.x + threadIdx.x;

        if (index < size) {
            out[index] += pow(a[index], power);
        }
    }


    template<class T>
    __global__ void subtract_power_device(T* out, const T* a, double power, size_t size) {
        size_t index = blockIdx.x*blockDim.x + threadIdx.x;

        if (index < size) {
            out[index] -= pow(a[index], power);
        }
    }

    template<class T>
    __global__ void compute_total_variation_device(T* out, const T* data, size_t nx, size_t ny, size_t nz, int p) {
        const auto coordinates = alsfvm::cuda::getCoordinates(threadIdx, blockIdx, blockDim,
                       nx - 1, ny > 1 ? ny - 1 : ny , nz > 1 ? nz - 1 : nz, {1, ny > 1, nz>1});


        if (coordinates.x < 0) {
            return;
        }

        const int x = coordinates.x;
        const int y = coordinates.y;
        const int z = coordinates.z;

        const size_t index = z * nx * ny + y * nx + x;
        const size_t indexXLeft = z * nx * ny + y * nx + (x-1);

        const size_t yBottom = ny > 0 ? y - 1 : 0;

        const size_t indexYLeft = z * nx * ny + yBottom * nx + x;
        const size_t indexLeft = z * nx * ny + yBottom * nx + (x-1);


        out[index] = pow(sqrt(pow(data[index]
                - data[indexXLeft],2) + pow( data[index]
                - data[indexYLeft],2)),p);

    }


    template<class T>
    __global__ void compute_total_variation_device(T* out, const T* data, size_t nx, size_t ny, size_t nz, size_t direction, int p) {

        const auto directionVector = alsfvm::make_direction_vector(direction);
        const auto coordinates = alsfvm::cuda::getCoordinates(threadIdx, blockIdx, blockDim,
                                                              nx - directionVector.x,
                                                              ny - directionVector.y,
                                                              nz - directionVector.z,
                                                              directionVector);


        if (coordinates.x < 0) {
            return;
        }

        const int x = coordinates.x;
        const int y = coordinates.y;
        const int z = coordinates.z;

        const size_t index = z * nx * ny + y * nx + x;


        const auto coordinatesLeft = coordinates - directionVector;


        const size_t indexLeft = coordinatesLeft.z*nx*ny + coordinatesLeft.y * nx + coordinatesLeft.x;

        out[index] = pow(fabs(data[index] - data[indexLeft]),p);
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

        template<class T>
        void add_power(T* a, const T* b, double power, size_t size) {
            const size_t threadCount = 1024;
            add_power_device << <(size + threadCount - 1) / threadCount, threadCount >> >(a, b, power, size);
        }


        template<class T>
        void subtract_power(T* a, const T* b, double power, size_t size) {
            const size_t threadCount = 1024;
            subtract_power_device << <(size + threadCount - 1) / threadCount, threadCount >> >(a, b, power, size);
        }

        template<class T>
        T compute_total_variation(const T* a, size_t nx, size_t ny, size_t nz, int p) {
            thrust::device_vector<T> buffer(nx*ny*nz, 0);

            if (nz > 1) {
                THROW("Only supported for 2d and 1d");
            }

            auto launchParameters = cuda::makeKernelLaunchParameters(ivec3{1, ny>1, nz>1},
                                                                    ivec3{nx, ny, nz},
                                                                    1024);


            compute_total_variation_device<<<std::get<0>(launchParameters), 1024>>>(thrust::raw_pointer_cast(buffer.data()),
                                                                             a, nx, ny, nz, p);

            return thrust::reduce(buffer.begin(), buffer.end());



        }


        template<class T>
        T compute_total_variation(const T* a, size_t nx, size_t ny, size_t nz, size_t direction, int p) {
            thrust::device_vector<T> buffer(nx*ny*nz, 0);

            if (nz > 1) {
                THROW("Only supported for 2d and 1d");
            }

            auto directionVector = make_direction_vector(direction);
            auto launchParameters = cuda::makeKernelLaunchParameters(directionVector,
                                                                    ivec3{nx, ny, nz},
                                                                    1024);


            compute_total_variation_device<<<std::get<0>(launchParameters), 1024>>>(thrust::raw_pointer_cast(buffer.data()),
                                                                             a, nx, ny, nz, direction, p);

            return thrust::reduce(buffer.begin(), buffer.end());



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

            template void add_power<real>(real* a, const real* b, double power, size_t size);

        template void subtract_power<real>(real* a, const real* b, double power, size_t size);

        template real compute_total_variation<real>(const real* a, size_t nx, size_t ny, size_t nz, int p);
        template real compute_total_variation<real>(const real* a, size_t nx, size_t ny, size_t nz, size_t direction, int p);


	}
}
