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

#include "alsfvm/cuda/CudaBuffer3D.hpp"
#include "cuda_runtime.h"
#include "alsfvm/cuda/cuda_utils.hpp"

namespace alsfvm {
namespace cuda {

template<class T> CudaBuffer3D<T>::CudaBuffer3D(size_t nx, size_t ny, size_t nz)
    : CudaBuffer<T>(nx * ny * nz * sizeof(T), nx, ny, nz, nx * sizeof(T),
          ny * sizeof(T)) {
    CUDA_SAFE_CALL(cudaMalloc(&memoryPointer, this->getSizeInBytes()));
}

template<class T> CudaBuffer3D<T>::~CudaBuffer3D() {
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