#include "alsfvm/memory/HostMemory.hpp"
#include "alsfvm/memory/memory_utils.hpp"
#include <cassert>
#include <algorithm>
#include "alsutils/error/Exception.hpp"
#define CHECK_SIZE_AND_HOST(x) { \
    if (!x.isOnHost()) {\
        THROW(#x << " is not on host."); \
    } \
    if (this->getSize() != x.getSize()) { \
        THROW("Size mismatch: \n\tthis->getSize() = " << this->getSize() <<"\n\t"<<#x<<".getSize() = " << x.getSize()); \
    } \
}
namespace alsfvm {
namespace memory {

template<class T>
HostMemory<T>::HostMemory(size_t nx, size_t ny, size_t nz)
    : Memory<T>(nx, ny, nz), data(nx*ny*nz, 42)
{

}

template<class T>
std::shared_ptr<Memory<T> > HostMemory<T>::makeInstance() const
{
    std::shared_ptr<Memory<T>> memoryArea;

    memoryArea.reset(new HostMemory(this->nx, this->ny, this->nz));

    return memoryArea;
}

template<class T>
bool HostMemory<T>::isOnHost() const
{
    return true;
}

template<class T>
void HostMemory<T>::copyFrom(const Memory<T> &other)
{
    CHECK_SIZE_AND_HOST(other);

    std::copy(other.data(), other.data() + this->getSize(), data.begin());
}

template<class T>
T *HostMemory<T>::getPointer()
{
    return data.data();
}

template<class T>
const T *HostMemory<T>::getPointer() const
{
	return data.data();
}

template<class T>
void HostMemory<T>::copyToHost(T *bufferPointer, size_t bufferLength) const
{
    assert(bufferLength >= Memory<T>::getSize());
    std::copy(data.begin(), data.end(), bufferPointer);
}

template<class T>
void HostMemory<T>::copyFromHost(const T* bufferPointer, size_t bufferLength)
{
    const size_t sizeToCopy = std::min(bufferLength, Memory<T>::getSize());
    std::copy(bufferPointer, bufferPointer+sizeToCopy, data.begin());
}



///
/// Adds the other memory area to this one
/// \param other the memory area to add from
///
template <class T>
void HostMemory<T>::operator+=(const Memory<T>& other) {
	if (!other.isOnHost()) {
		THROW("Memory not on host");
	}

	auto pointer = other.getPointer();
    #pragma omp parallel for simd
    for (int i = 0; i < int(data.size()); ++i) {
		data[i] += pointer[i];
	}
}

///
/// Mutliplies the other memory area to this one
/// \param other the memory area to multiply from
///
template <class T>
void HostMemory<T>::operator*=(const Memory<T>& other) {
	if (!other.isOnHost()) {
		THROW("Memory not on host");
	}

	if (other.getSize() != this->getSize()) {
		THROW("Memory size not the same");
	}

	auto pointer = other.getPointer();
    #pragma omp parallel for simd
    for (int i = 0; i < int(data.size()); ++i) {
		data[i] *= pointer[i];
	}
}

///
/// Subtracts the other memory area to this one
/// \param other the memory area to subtract from
///
template <class T>
void HostMemory<T>::operator-=(const Memory<T>& other) {
	if (!other.isOnHost()) {
		THROW("Memory not on host");
	}
	if (other.getSize() != this->getSize()) {
		THROW("Memory size not the same");
	}

	auto pointer = other.getPointer();
    #pragma omp parallel for simd
    for (int i = 0; i < int(data.size()); ++i) {
		data[i] -= pointer[i];
	}
}

///
/// Divides the other memory area to this one
/// \param other the memory area to divide from
///
template <class T>
void HostMemory<T>::operator/=(const Memory<T>& other) {
	if (!other.isOnHost()) {
		THROW("Memory not on host");
	}
	if (other.getSize() != this->getSize()) {
		THROW("Memory size not the same");
	}

	auto pointer = other.getPointer();
    #pragma omp parallel for simd
    for (int i = 0; i < int(data.size()); ++i) {
		data[i] /= pointer[i];
	}
}

///
/// Adds the scalar to each component
/// \param scalar the scalar to add
///
template <class T>
void HostMemory<T>::operator+=(real scalar) {

#pragma omp parallel for simd
    for (int i = 0; i < int(data.size()); ++i) {
		data[i] += scalar;
	}
}

///
/// Multiplies the scalar to each component
/// \param scalar the scalar to multiply
///
template <class T>
void HostMemory<T>::operator*=(real scalar) {
#pragma omp parallel for simd
    for (int i = 0; i < int(data.size()); ++i) {
		data[i] *= scalar;
	}
}

///
/// Subtracts the scalar from each component
/// \param scalar the scalar to subtract
///
template <class T>
void HostMemory<T>::operator-=(real scalar) {
#pragma omp parallel for simd
    for (int i = 0; i < int(data.size()); ++i) {
		data[i] -= scalar;
	}
}

///
/// Divides the each component by the scalar
/// \param scalar the scalar to divide
///
template <class T>
void HostMemory<T>::operator/=(real scalar) {
    #pragma omp parallel for simd
    for (int i = 0; i < int(data.size()); ++i) {
		data[i] /= scalar;
    }
}

template <class T>
void HostMemory<T>::makeZero()
{
    #pragma omp parallel for simd
    for (int i = 0; i < int(data.size()); ++i) {
        data[i] = 0;
    }
}


template <class T>
void HostMemory<T>::copyInternalCells(size_t startX, size_t endX, size_t startY, size_t endY, size_t startZ, size_t endZ, T *output, size_t outputSize)
{
    const size_t nx = this->nx;
    const size_t ny = this->ny;
    const size_t numberOfY = endY-startY;
    const size_t numberOfX = endX-startX;
    for(size_t z = startZ; z < endZ; z++) {
        for(size_t y = startY; y < endY; y++) {
            for(size_t x = startX; x < endX; x++) {
                size_t indexIn = z * nx * ny + y * nx + x;
                size_t indexOut = (z-startZ) * numberOfX * numberOfY
                      + (y - startY) * numberOfX + (x - startX);
                output[indexOut] = data[indexIn];
             }
        }
    }
}

template<class T>
void HostMemory<T>::addLinearCombination(T a1,
    T a2, const Memory<T>& v2,
    T a3, const Memory<T>& v3,
    T a4, const Memory<T>& v4,
    T a5, const Memory<T>& v5) {
    CHECK_SIZE_AND_HOST(v2);
    CHECK_SIZE_AND_HOST(v3);
    CHECK_SIZE_AND_HOST(v4);
    CHECK_SIZE_AND_HOST(v5);
    const auto& d1 = data;
    auto d2 = v2.getPointer();
    auto d3 = v3.getPointer();
    auto d4 = v4.getPointer();
    auto d5 = v5.getPointer();
#pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = a1*d1[i] + a2*d2[i] + a3*d3[i] + a4*d4[i] + a5*d5[i];
    }
}

template<class T>
void HostMemory<T>::addPower(const Memory<T> &other, double power)
{
    CHECK_SIZE_AND_HOST(other);
    #pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] += std::pow(other[i], power);
    }
}

template<class T>
void HostMemory<T>::subtractPower(const Memory<T> &other, double power)
{
    CHECK_SIZE_AND_HOST(other);
    #pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] -= std::pow(other[i], power);
    }
}

template<class T>
std::shared_ptr<Memory<T> > HostMemory<T>::getHostMemory()
{
    return this->shared_from_this();
}

template<class T>
real HostMemory<T>::getTotalVariation(int p) const
{
    // See http://www.ams.org/journals/tran/1933-035-04/S0002-9947-1933-1501718-2/S0002-9947-1933-1501718-2.pdf
    //
    const size_t nx = this->nx;
    const size_t ny = this->ny;
    const size_t nz = this->nz;

    if (nz > 1 ) {
        THROW("Not supported for 3d yet");
    }
    const size_t startX = 1;
    const size_t startY = ny > 1 ? 1 : 0;
    T bv = 0;
    for(size_t z = 0; z < nz; z++) {
        for(size_t y = startY; y < ny; y++) {
            for(size_t x = startX; x < nx; x++) {
                size_t index = z * nx * ny + y * nx + x;
                size_t indexXLeft = z * nx * ny + y * nx + (x-1);

                size_t yBottom = ny > 0 ? y - 1 : 0;


                size_t indexYLeft = z * nx * ny + yBottom * nx + x;
                size_t indexLeft = z * nx * ny + yBottom * nx + (x-1);

                bv += std::pow(std::sqrt(std::pow(data[index]
                        - data[indexYLeft],2) + std::pow(data[index]
                        - data[indexXLeft],2)), p);
             }
        }
    }

    return bv;

}

template<class T>
real HostMemory<T>::getTotalVariation(int direction, int p) const
{
    // See http://www.ams.org/journals/tran/1933-035-04/S0002-9947-1933-1501718-2/S0002-9947-1933-1501718-2.pdf
    //

    auto directionVector = make_direction_vector(direction);
    const size_t nx = this->nx;
    const size_t ny = this->ny;
    const size_t nz = this->nz;

    if (direction > (1+(ny>1)+(nz>1))) {
        THROW("direction = " << direction << " is bigger than current dimension");
    }
    const size_t startX = directionVector.x;
    const size_t startY = directionVector.y;
    const size_t startZ = directionVector.z;
    T bv = 0;

    const auto view = this->getView();


    for(size_t z = startZ; z < nz; z++) {
        for(size_t y = startY; y < ny; y++) {
            for(size_t x = startX; x < nx; x++) {
                size_t index = z * nx * ny + y * nx + x;
                auto positionLeft = ivec3(x,y,z)-directionVector;

                bv += std::pow(std::abs(view.at(x,y,z)-view.at(positionLeft.x,positionLeft.y, positionLeft.z)), p);
             }
        }
    }

    return bv;

}


INSTANTIATE_MEMORY(HostMemory)
} // namespace memory
} // namespace alsfvm

