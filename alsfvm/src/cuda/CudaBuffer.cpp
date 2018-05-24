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

#include "alsfvm/cuda/CudaBuffer.hpp"

namespace alsfvm {
namespace cuda {

template<class T> CudaBuffer<T>::CudaBuffer(size_t sizeInBytes,
    size_t nx,
    size_t ny,
    size_t nz,
    size_t pitchX,
    size_t pitchY)
    : sizeInBytes(sizeInBytes), nx(nx), ny(ny), nz(nz), pitchX(pitchX),
      pitchY(pitchY) {
    // Empty
}

/// Virtual because we need to inherit from this
template<class T> CudaBuffer<T>::~CudaBuffer() {
    // Empty
}


///
/// \returns the total size of the memory area in bytes
///
template<class T>
size_t CudaBuffer<T>::getSizeInBytes() const {
    return sizeInBytes;
}

///
/// \returns the size (in number of T's)
///
template<class T>
size_t CudaBuffer<T>::getSize() const {
    return nx * ny * nz;
}

///
/// \note Do not use this for indexing!
/// \returns the number of T's in X direction
///
template<class T>
size_t CudaBuffer<T>::getXSize() const {
    return nx;
}


///
/// \note Do not use this for indexing!
/// \returns the number of T's in Y direction
///
template<class T>
size_t CudaBuffer<T>::getYSize() const {
    return ny;
}


///
/// \note Do not use this for indexing!
/// \returns the number of T's in Z direction
///
template<class T>
size_t CudaBuffer<T>::getZSize() const {
    return nz;
}

///
/// \returns the pitch in Y direction  (in bytes!)
/// \note Use for indexing by
/// \code{.cpp}
/// size_t index = i*getPitchY()*getPitchX() + j*getPitchX() + k;
/// \endcode
///
template<class T>
size_t CudaBuffer<T>::getPitchY() const {
    return pitchY;
}


///
/// \returns the pitch in X direction (in bytes!)
/// \note Use for indexing by
/// \code{.cpp}
/// size_t index = i*getPitchY()*getPitchX() + j*getPitchX() + k;
/// \endcode
///
template<class T>
size_t CudaBuffer<T>::getPitchX() const {
    return pitchX;
}
}
}