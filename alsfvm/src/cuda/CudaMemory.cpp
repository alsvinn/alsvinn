#include "alsfvm/cuda/CudaMemory.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include "alsfvm/cuda/cuda_utils.hpp"
#include <cassert>
#include <algorithm>
#include "alsfvm/memory/memory_utils.hpp"
#include "alsfvm/cuda/vector_operations.hpp"
#include "alsfvm/error/Exception.hpp"

namespace alsfvm {
	namespace cuda {

		template<class T>
		CudaMemory<T>::CudaMemory(size_t size) 
			: memory::Memory<T>(size) 
		{
			CUDA_SAFE_CALL(cudaMalloc(&memoryPointer, size*sizeof(T)));
		}

		// Note: Virtual distructor since we will inherit
		// from this. 
		template<class T>
		CudaMemory<T>::~CudaMemory() 
		{
			CUDA_SAFE_CALL(cudaFree(memoryPointer));
		}

		

		///
		/// Checks if the memory area is on the host (CPU) or 
		/// on some device, if the latter, one needs to copy to host
		/// before reading it.
		/// @returns true if the memory is on host, false otherwise
		///
		template<class T>
		bool CudaMemory<T>::isOnHost() const
		{
			return false;
		}

		///
		/// Gets the pointer to the data (need not be on the host!)
		///
		template<class T>
		T* CudaMemory<T>::getPointer() {
			return memoryPointer;
		}

		///
		/// Gets the pointer to the data (need not be on the host!)
		///
		template<class T>
		const T* CudaMemory<T>::getPointer() const {
			return memoryPointer;
		}

		/// 
		/// Copies the memory to the given buffer
		///
		template<class T>
		void CudaMemory<T>::copyToHost(T* bufferPointer, size_t bufferLength) {
			assert(bufferLength >= size);
			CUDA_SAFE_CALL(cudaMemcpy(bufferPointer, memoryPointer, size*sizeof(T), cudaMemcpyDeviceToHost));
		}

		///
		/// Copies the memory from the buffer (assumed to be on Host/CPU)
		///
		template<class T>
		void CudaMemory<T>::copyFromHost(const T* bufferPointer, size_t bufferLength) {
			const size_t copySize = std::min(bufferLength, size);
			CUDA_SAFE_CALL(cudaMemcpy(memoryPointer, bufferPointer, copySize*sizeof(T), cudaMemcpyHostToDevice));
		}


		///
		/// Adds the other memory area to this one
		/// \param other the memory area to add from
		///
		template<class T>
		void CudaMemory<T>::operator+=(const Memory<T>& other) {
			if (other.getSize() != this->getSize()) {
				THROW("Memory size not the same");
			}

			add(getPointer(), getPointer(),
				other.getPointer(), Memory<T>::getSize());
		}


		///
		/// Mutliplies the other memory area to this one
		/// \param other the memory area to multiply from
		///
		template<class T>
		void CudaMemory<T>::operator*=(const Memory<T>& other) {
			if (other.getSize() != this->getSize()) {
				THROW("Memory size not the same");
			}

			multiply(getPointer(), getPointer(),
				other.getPointer(), Memory<T>::getSize());
		}

		///
		/// Subtracts the other memory area to this one
		/// \param other the memory area to subtract from
		///
		template<class T>
		void CudaMemory<T>::operator-=(const Memory<T>& other) {
			if (other.getSize() != this->getSize()) {
				THROW("Memory size not the same");
			}

			subtract(getPointer(), getPointer(),
				other.getPointer(), Memory<T>::getSize());
		}

		///
		/// Divides the other memory area to this one
		/// \param other the memory area to divide from
		///
		template<class T>
		void CudaMemory<T>::operator/=(const Memory<T>& other) {
			if (other.getSize() != this->getSize()) {
				THROW("Memory size not the same");
			}

			divide(getPointer(), getPointer(),
				other.getPointer(), Memory<T>::getSize());
		}


		///
		/// Adds the scalar to each component
		/// \param scalar the scalar to add
		///
		template<class T>
		void CudaMemory<T>::operator+=(real scalar) {
			add(getPointer(), getPointer(),
				scalar, Memory<T>::getSize());
		}
		///
		/// Multiplies the scalar to each component
		/// \param scalar the scalar to multiply
		///
		template<class T>
		void CudaMemory<T>::operator*=(real scalar) {
			multiply(getPointer(), getPointer(),
				scalar, Memory<T>::getSize());
		}

		///
		/// Subtracts the scalar from each component
		/// \param scalar the scalar to subtract
		///
		template<class T>
		void CudaMemory<T>::operator-=(real scalar) {
			subtract(getPointer(), getPointer(),
				scalar, Memory<T>::getSize());
		}

		///
		/// Divides the each component by the scalar
		/// \param scalar the scalar to divide
		///
		template<class T>
		void CudaMemory<T>::operator/=(real scalar) {
			divide(getPointer(), getPointer(),
				scalar, Memory<T>::getSize());
		}

		INSTANTIATE_MEMORY(CudaMemory)
		ADD_MEMORY_TO_FACTORY(CudaMemory)
	}

}