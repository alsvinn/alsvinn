#include "alsfvm/memory/Memory.hpp"
#include "alsfvm/memory/memory_utils.hpp"

namespace alsfvm {
	namespace memory {
        template<class T>
        Memory<T>::Memory(size_t nx, size_t ny, size_t nz)
            : nx(nx), ny(ny), nz(nz)
        {

        }

        template<class T>
        size_t Memory<T>::getSize() const {
            return nx*ny*nz;
        }

        template<class T>
        size_t Memory<T>::getSizeX() const
        {
            return nx;
        }

        template<class T>
        size_t Memory<T>::getSizeY() const
        {
            return ny;
        }

        template<class T>
        size_t Memory<T>::getSizeZ() const
        {
            return nz;
        }

        template<class T>
        size_t Memory<T>::getExtentXInBytes() const
        {
            return nx*sizeof(T);
        }

        template<class T>
        size_t Memory<T>::getExtentYInBytes() const
        {
            return ny*sizeof(T);
        }

        INSTANTIATE_MEMORY(Memory)
	}
}
