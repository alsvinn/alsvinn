#include "alsfvm/cuda/CudaMemory.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include "alsfvm/cuda/cuda_utils.hpp"
#include <cassert>
#include <algorithm>
#include "alsfvm/memory/memory_utils.hpp"

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
		bool CudaMemory<T>::isOnHost() 
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

		INSTANTIATE_MEMORY(CudaMemory)
	}

}