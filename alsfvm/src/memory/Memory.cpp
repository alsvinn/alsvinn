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

#include "alsfvm/memory/Memory.hpp"
#include "alsfvm/memory/memory_utils.hpp"

namespace alsfvm {
namespace memory {
template<class T> Memory<T>::Memory(size_t nx, size_t ny, size_t nz)
    : nx(nx), ny(ny), nz(nz) {

}

template<class T>
size_t Memory<T>::getSize() const {
    return nx * ny * nz;
}

template<class T>
size_t Memory<T>::getSizeX() const {
    return nx;
}

template<class T>
size_t Memory<T>::getSizeY() const {
    return ny;
}

template<class T>
size_t Memory<T>::getSizeZ() const {
    return nz;
}

template<class T>
size_t Memory<T>::getExtentXInBytes() const {
    return nx * sizeof(T);
}

template<class T>
size_t Memory<T>::getExtentYInBytes() const {
    return ny * sizeof(T);
}

template<class T>
View<T> Memory<T>::getView() {
    return View<T>(getPointer(), nx, ny, nz, getExtentXInBytes(),
            getExtentYInBytes());
}

template<class T>
View<const T> Memory<T>::getView() const {
    return View<const T>(getPointer(), nx, ny, nz, getExtentXInBytes(),
            getExtentYInBytes());
}



INSTANTIATE_MEMORY(Memory)
}
}
