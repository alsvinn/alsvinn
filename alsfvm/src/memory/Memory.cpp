#include "alsfvm/memory/Memory.hpp"
namespace alsfvm {
	namespace memory {
		Memory::Memory(size_t size) : size(size) {}

		size_t Memory::getSize() {
			return size;
		}
	}
}