#include "alsfvm/cuda/CudaBuffer.hpp"

namespace alsfvm {
namespace cuda {

template<class T>
CudaBuffer<T>::CudaBuffer(size_t sizeInBytes,
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
template<class T>
CudaBuffer<T>::~CudaBuffer() {
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