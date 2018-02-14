#include "alsfvm/cuda/CudaBuffer3D.hpp"
#include "cuda_runtime.h"
#include "alsfvm/cuda/cuda_utils.hpp"

namespace alsfvm {
namespace cuda {

template<class T>
CudaBuffer3D<T>::CudaBuffer3D(size_t nx, size_t ny, size_t nz)
    : CudaBuffer<T>(nx * ny * nz * sizeof(T), nx, ny, nz, nx * sizeof(T),
          ny * sizeof(T)) {
    CUDA_SAFE_CALL(cudaMalloc(&memoryPointer, this->getSizeInBytes()));
}

template<class T>
CudaBuffer3D<T>::~CudaBuffer3D() {
    CUDA_SAFE_CALL(cudaFree(memoryPointer));
}

template<class T>
T* CudaBuffer3D<T>::getPointer() {
    return memoryPointer;
}

template<class T>
const T* CudaBuffer3D<T>::getPointer() const {
    return memoryPointer;
}
}
}