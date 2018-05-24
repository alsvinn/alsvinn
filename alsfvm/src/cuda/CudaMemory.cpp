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

#include "alsfvm/cuda/CudaMemory.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include "alsfvm/cuda/cuda_utils.hpp"
#include <cassert>
#include <algorithm>
#include "alsfvm/memory/memory_utils.hpp"
#include "alsfvm/cuda/vector_operations.hpp"
#include "alsutils/error/Exception.hpp"
#include "alsfvm/memory/HostMemory.hpp"
#include "alsutils/log.hpp"

#define CHECK_SIZE_AND_HOST(x) { \
    if (x.isOnHost()) {\
        THROW(#x << " is on host."); \
    } \
    if (this->getSize() != x.getSize()) { \
        THROW("Size mismatch: \n\tthis->getSize() = " << this->getSize() <<"\n\t"<<#x<<".getSize() = " << x.getSize()); \
    } \
}
using namespace alsfvm::memory;
namespace alsfvm {
namespace cuda {

template<class T> CudaMemory<T>::CudaMemory(size_t nx, size_t ny, size_t nz)
    : memory::Memory<T>(nx, ny, nz) {
    CUDA_SAFE_CALL(cudaMalloc(&memoryPointer, nx * ny * nz * sizeof(T)));
    CUDA_SAFE_CALL(cudaMemset(memoryPointer, 0, nx * ny * nz * sizeof(T)));
}

template<class T>
std::shared_ptr<memory::Memory<T> > CudaMemory<T>::makeInstance() const {
    std::shared_ptr<memory::Memory<T> > memoryArea;
    memoryArea.reset(new CudaMemory<T>(this->nx, this->ny, this->nz));

    return memoryArea;
}

// Note: Virtual distructor since we will inherit
// from this.
template<class T> CudaMemory<T>::~CudaMemory() {
    // We do not want to throw exceptions from a Destructor, see
    // http://en.cppreference.com/w/cpp/language/destructor#Exceptions
    try {
        CUDA_SAFE_CALL(cudaFree(memoryPointer));
    } catch (std::runtime_error& e) {
        ALSVINN_LOG(ERROR, "Could not delete CudaMemory, error message was\n\n"
            << e.what());
    }
}

template<class T>
void CudaMemory<T>::copyFrom(const memory::Memory<T>& other) {

    if (other.isOnHost()) {
        this->copyFromHost(other.getPointer(), other.getSize());
    } else {
        CHECK_SIZE_AND_HOST(other);

        CUDA_SAFE_CALL(cudaMemcpy(memoryPointer, other.data(),
                this->nx * this->ny * this->nz * sizeof(T),
                cudaMemcpyDeviceToDevice));
    }
}



///
/// Checks if the memory area is on the host (CPU) or
/// on some device, if the latter, one needs to copy to host
/// before reading it.
/// @returns true if the memory is on host, false otherwise
///
template<class T>
bool CudaMemory<T>::isOnHost() const {
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
void CudaMemory<T>::copyToHost(T* bufferPointer, size_t bufferLength) const {
    assert(bufferLength >= this->getSize());
    CUDA_SAFE_CALL(cudaMemcpy(bufferPointer, memoryPointer,
            this->getSize()*sizeof(T), cudaMemcpyDeviceToHost));
}

///
/// Copies the memory from the buffer (assumed to be on Host/CPU)
///
template<class T>
void CudaMemory<T>::copyFromHost(const T* bufferPointer, size_t bufferLength) {
    const size_t copySize = std::min(bufferLength, this->getSize());
    CUDA_SAFE_CALL(cudaMemcpy(memoryPointer, bufferPointer, copySize * sizeof(T),
            cudaMemcpyHostToDevice));
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

template<class T>
void CudaMemory<T>::makeZero() {
    CUDA_SAFE_CALL(cudaMemset(getPointer(), 0, Memory<T>::getSize()*sizeof(T)));
}

template<class T>
void CudaMemory<T>::copyInternalCells(size_t startX, size_t endX,
    size_t startY, size_t endY,
    size_t startZ, size_t endZ,
    T* output, size_t outputSize) {

    // Until we start using cudamalloc3d, we need to do this
    // by first copying the memory to cpu, then reformatting it on cpu
    std::vector<T> temporaryStorage(this->getSize());

    copyToHost(temporaryStorage.data(), temporaryStorage.size());

    const size_t nx = this->nx;
    const size_t ny = this->ny;
    const size_t numberOfY = endY - startY;
    const size_t numberOfX = endX - startX;

    for (size_t z = startZ; z < endZ; z++) {
        for (size_t y = startY; y < endY; y++) {
            for (size_t x = startX; x < endX; x++) {
                size_t indexIn = z * nx * ny + y * nx + x;
                size_t indexOut = (z - startZ) * numberOfX * numberOfY
                    + (y - startY) * numberOfX + (x - startX);
                output[indexOut] = temporaryStorage[indexIn];
            }
        }
    }
}

template<class T>
void CudaMemory<T>::addLinearCombination(T a1,
    T a2, const memory::Memory<T>& v2,
    T a3, const memory::Memory<T>& v3,
    T a4, const memory::Memory<T>& v4,
    T a5, const memory::Memory<T>& v5) {
    CHECK_SIZE_AND_HOST(v2);
    CHECK_SIZE_AND_HOST(v3);
    CHECK_SIZE_AND_HOST(v4);
    CHECK_SIZE_AND_HOST(v5);
    auto d1 = getPointer();
    auto d2 = v2.getPointer();
    auto d3 = v3.getPointer();
    auto d4 = v4.getPointer();
    auto d5 = v5.getPointer();

    add_linear_combination(a1, d1,
        a2, d2,
        a3, d3,
        a4, d4,
        a5, d5,
        this->getSize());

}

template<class T>
void CudaMemory<T>::addPower(const memory::Memory<T>& other, double power) {
    CHECK_SIZE_AND_HOST(other);
    add_power(getPointer(), other.getPointer(), power, this->getSize());
}


template<class T>
void CudaMemory<T>::subtractPower(const memory::Memory<T>& other,
    double power) {
    CHECK_SIZE_AND_HOST(other);
    subtract_power(getPointer(), other.getPointer(), power, this->getSize());
}

template<class T>
std::shared_ptr<memory::Memory<T> > CudaMemory<T>::getHostMemory() {
    std::shared_ptr<memory::Memory<T> > pointer(new memory::HostMemory<T>(this->nx,
            this->ny, this->nz));

    this->copyToHost(pointer->getPointer(), pointer->getSize());

    return pointer;
}

template<class T>
real CudaMemory<T>::getTotalVariation(int p, const ivec3& start,
    const ivec3& end) const {
    return compute_total_variation(this->getPointer(), this->nx,
            this->ny,
            this->nz, p, start, end);
}

template<class T>
real CudaMemory<T>::getTotalVariation(int direction, int p, const ivec3& start,
    const ivec3& end) const {
    return compute_total_variation(this->getPointer(), this->nx,
            this->ny,
            this->nz, direction, p, start, end);
}

INSTANTIATE_MEMORY(CudaMemory)
}

}
