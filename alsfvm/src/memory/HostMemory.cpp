#include "alsfvm/memory/HostMemory.hpp"
#include "alsfvm/memory/memory_utils.hpp"
#include <cassert>
namespace alsfvm {
namespace memory {

template<class T>
HostMemory<T>::HostMemory(size_t size)
    : Memory<T>(size), data(size)
{

}

template<class T>
bool HostMemory<T>::isOnHost()
{
    return true;
}

template<class T>
T *HostMemory<T>::getPointer()
{
    return data.data();
}

template<class T>
void HostMemory<T>::copyToHost(T *bufferPointer, size_t bufferLength)
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

INSTANTIATE_MEMORY(HostMemory)
} // namespace memory
} // namespace alsfvm

