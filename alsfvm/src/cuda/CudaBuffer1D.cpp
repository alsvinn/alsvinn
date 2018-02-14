#include "alsfvm/cuda/CudaBuffer1D.hpp"
#include "cuda_runtime.h"
#include "alsfvm/cuda/cuda_utils.hpp"

namespace alsfvm {
namespace cuda {

template<class T> CudaBuffer1D<T>::CudaBuffer1D(size_t nx)
    : CudaBuffer<T>(nx * sizeof(T), nx, 1, 1, nx * sizeof(T), 1) {
    CUDA_SAFE_CALL(cudaMalloc(&memoryPointer, this->getSizeInBytes()));
}

template<class T> CudaBuffer1D<T>::~CudaBuffer1D() {
    CUDA_SAFE_CALL(cudaFree(memoryPointer));
}

template<class T>
T* CudaBuffer1D<T>::getPointer() {
    return memoryPointer;
}

template<class T>
const T* CudaBuffer1D<T>::getPointer() const {
    return memoryPointer;
}
}
}