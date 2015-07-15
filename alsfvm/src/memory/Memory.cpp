#include "alsfvm/memory/Memory.hpp"
#include "alsfvm/memory/memory_utils.hpp"

namespace alsfvm {
	namespace memory {
        template<class T>
        Memory<T>::Memory(size_t size) : size(size) {}

        template<class T>
        size_t Memory<T>::getSize() {
			return size;
		}

        INSTANTIATE_MEMORY(Memory)
	}
}
