#pragma once
#include <vector>
#include "alsfvm/memory/Memory.hpp"
#include "alsfvm/memory/index.hpp"

namespace alsfvm {
namespace memory {

template<class T>
class HostMemory : public alsfvm::memory::Memory<T>
{
public:

    ///
    /// @param size the size of the memory area (in bytes)
    ///
    HostMemory(size_t nx, size_t ny=1, size_t nz=1);


    ///
    /// Checks if the memory area is on the host (CPU) or
    /// on some device, if the latter, one needs to copy to host
    /// before reading it.
    /// @returns true if the memory is on host, false otherwise
    ///
    virtual bool isOnHost() const;

    ///
    /// Gets the pointer to the data (need not be on the host!)
    /// \note If this is an OpenCL implementation, the pointer will
    /// be useless! If you want to use the OpenCL memory, you should
    /// first cast to OpenCL memory, then get the OpenCL buffer pointer.
    ///
    virtual T* getPointer();

	///
	/// Gets the pointer to the data (need not be on the host!)
	/// \note If this is an OpenCL implementation, the pointer will
	/// be useless! If you want to use the OpenCL memory, you should
	/// first cast to OpenCL memory, then get the OpenCL buffer pointer.
	///
	virtual const T* getPointer() const;

    ///
    /// Copies the memory to the given buffer
    /// \note bufferLength must be at least getSize()
    /// \param bufferPointer the buffer to write to
    /// \param bufferLength the size of the buffer (in number of T's)
    ///
    virtual void copyToHost(T* bufferPointer,
                            size_t bufferLength);

    ///
    /// Copies the memory from the buffer (assumed to be on Host/CPU)
    /// \note bufferLength must be at least getSize()
    /// \param bufferPointer the buffer to write to
    /// \param bufferLength the size of the buffer (in number of T's)
    ///
    virtual void copyFromHost(const T* bufferPointer,
                              size_t bufferLength);


	///
	/// Adds the other memory area to this one
	/// \param other the memory area to add from
	///
	virtual void operator+=(const Memory<T>& other);

	///
	/// Mutliplies the other memory area to this one
	/// \param other the memory area to multiply from
	///
	virtual void operator*=(const Memory<T>& other);

	///
	/// Subtracts the other memory area to this one
	/// \param other the memory area to subtract from
	///
	virtual void operator-=(const Memory<T>& other);

	///
	/// Divides the other memory area to this one
	/// \param other the memory area to divide from
	///
	virtual void operator/=(const Memory<T>& other);

	///
	/// Adds the scalar to each component
	/// \param scalar the scalar to add
	///
	virtual void operator+=(real scalar);

	///
	/// Multiplies the scalar to each component
	/// \param scalar the scalar to multiply
	///
	virtual void operator*=(real scalar);

	///
	/// Subtracts the scalar from each component
	/// \param scalar the scalar to subtract
	///
	virtual void operator-=(real scalar);

	///
	/// Divides the each component by the scalar
	/// \param scalar the scalar to divide
	///
	virtual void operator/=(real scalar);


    virtual T& at(size_t x, size_t y=0, size_t z=0);

    virtual const T& at(size_t x, size_t y=0, size_t z=0) const;


private:
    std::vector<T> data;
};

template<class T>
T& HostMemory<T>::at(size_t x, size_t y, size_t z) {
    return data[calculateIndex(x, y, z, this->nx, this->ny)];
}

template<class T>
const T& HostMemory<T>::at(size_t x, size_t y, size_t z) const {
    return data[calculateIndex(x, y, z, this->nx, this->ny)];
}

} // namespace memory
} // namespace alsfvm


