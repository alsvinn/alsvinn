/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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