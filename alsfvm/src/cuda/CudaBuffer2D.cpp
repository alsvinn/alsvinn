#include "alsfvm/cuda/CudaBuffer2D.hpp"
#include "cuda_runtime.h"
#include "alsfvm/cuda/cuda_utils.hpp"

namespace alsfvm {
namespace cuda {

template<class T> CudaBuffer2D<T>::CudaBuffer2D(size_t nx, size_t ny)
    : CudaBuffer<T>(nx * ny * sizeof(T), nx, ny, 1, nx * sizeof(T),
          ny * sizeof(T)) {
    CUDA_SAFE_CALL(cudaMalloc(&memoryPointer, this->getSizeInBytes()));
}

template<class T> CudaBuffer2D<T>::~CudaBuffer2D() {
    CUDA_SAFE_CALL(cudaFree(memoryPointer));
}

template<class T>
T* CudaBuffer2D<T>::getPointer() {
    return memoryPointer;
}

template<class T>
const T* CudaBuffer2D<T>::getPointer() const {
    return memoryPointer;
}
}
}