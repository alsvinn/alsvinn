#include "alsfvm/memory/HostMemory.hpp"
#include "alsfvm/memory/memory_utils.hpp"
#include <cassert>
#include <algorithm>
#include "alsfvm/error/Exception.hpp"

namespace alsfvm {
namespace memory {

template<class T>
HostMemory<T>::HostMemory(size_t size)
    : Memory<T>(size), data(size)
{

}

template<class T>
bool HostMemory<T>::isOnHost() const
{
    return true;
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
	for (size_t i = 0; i < data.size(); ++i) {
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

	auto pointer = other.getPointer();
	for (size_t i = 0; i < data.size(); ++i) {
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

	auto pointer = other.getPointer();
	for (size_t i = 0; i < data.size(); ++i) {
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

	auto pointer = other.getPointer();
	for (size_t i = 0; i < data.size(); ++i) {
		data[i] /= pointer[i];
	}
}

INSTANTIATE_MEMORY(HostMemory)
ADD_MEMORY_TO_FACTORY(HostMemory)
} // namespace memory
} // namespace alsfvm

