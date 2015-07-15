#pragma once
#include <vector>
#include "alsfvm/memory/Memory.hpp"
namespace alsfvm {
namespace memory {

template<class T>
class HostMemory : public alsfvm::memory::Memory<T>
{
public:

    ///
    /// @param size the size of the memory area (in bytes)
    ///
    HostMemory(size_t size);


    ///
    /// Checks if the memory area is on the host (CPU) or
    /// on some device, if the latter, one needs to copy to host
    /// before reading it.
    /// @returns true if the memory is on host, false otherwise
    ///
    virtual bool isOnHost();

    ///
    /// Gets the pointer to the data (need not be on the host!)
    /// \note If this is an OpenCL implementation, the pointer will
    /// be useless! If you want to use the OpenCL memory, you should
    /// first cast to OpenCL memory, then get the OpenCL buffer pointer.
    ///
    virtual T* getPointer();

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
private:
    std::vector<T> data;
};

} // namespace memory
} // namespace alsfvm


