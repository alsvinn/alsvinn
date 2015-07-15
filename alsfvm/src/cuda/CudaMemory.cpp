#include "alsfvm/cuda/CudaMemory.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include "alsfvm/cuda/cuda_utils.hpp"
#include <cassert>
#include <algorithm>

namespace alsfvm {
	namespace cuda {

		CudaMemory::CudaMemory(size_t size) 
			: memory::Memory(size) 
		{
			CUDA_SAFE_CALL(cudaMalloc(&memoryPointer, size));
		}

		// Note: Virtual distructor since we will inherit
		// from this. 
		CudaMemory::~CudaMemory() 
		{
			CUDA_SAFE_CALL(cudaFree(memoryPointer));
		}

		

		///
		/// Checks if the memory area is on the host (CPU) or 
		/// on some device, if the latter, one needs to copy to host
		/// before reading it.
		/// @returns true if the memory is on host, false otherwise
		///
		bool CudaMemory::isOnHost() 
		{
			return false;
		}

		///
		/// Gets the pointer to the data (need not be on the host!)
		///
		void* CudaMemory::getPointer() {
			return memoryPointer;
		}

		/// 
		/// Copies the memory to the given buffer
		///
		void CudaMemory::copyToHost(void* bufferPointer, size_t bufferLength) {
			assert(bufferLength >= size);
			CUDA_SAFE_CALL(cudaMemcpy(bufferPointer, memoryPointer, size, cudaMemcpyDeviceToHost));
		}

		///
		/// Copies the memory from the buffer (assumed to be on Host/CPU)
		///
		void CudaMemory::copyFromHost(const void* bufferPointer, size_t bufferLength) {
			const size_t copySize = std::min(bufferLength, size);
			CUDA_SAFE_CALL(cudaMemcpy(memoryPointer, bufferPointer, copySize, cudaMemcpyHostToDevice));
		}
	}
}