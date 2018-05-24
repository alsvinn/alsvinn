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

#pragma once

///
/// This file contains various utility functions for indexing
///

///
/// \brief dataAtRaw gets the data at the given indexes
/// \param pointer the pointer to the data
/// \param ix the x index (in bytes)
/// \param iy the y index (in bytes)
/// \param iz the z index (in bytes)
/// \param nx the number of elements in x direction  (in bytes)
/// \param ny the number of elements in z direction  (in bytes)
/// \return the pointer to the data element
///
inline const char* dataAtRawConst(const char* pointer,
    size_t ix, size_t iy, size_t iz, size_t nx, size_t ny) {

    return pointer + iz * nx * ny + iy * nx + ix;
}

///
/// \brief dataAtRaw gets the data at the given indexes
/// \param pointer the pointer to the data
/// \param ix the x index (in bytes)
/// \param iy the y index (in bytes)
/// \param iz the z index (in bytes)
/// \param nx the number of elements in x direction  (in bytes)
/// \param ny the number of elements in z direction  (in bytes)
/// \return the pointer to the data element
///
inline char* dataAtRaw(char* pointer,
    size_t ix, size_t iy, size_t iz, size_t nx, size_t ny) {

    return pointer + iz * nx * ny + iy * nx + ix;
}

///
/// \brief dataAt gets the data at the given address
/// \param pointer the pointer to the data
/// \param ix the x index
/// \param iy the y index
/// \param iz the z index
/// \param nx the number of elements in x direction  (in bytes)
/// \param ny the number of elements in z direction  (in bytes)
/// \return the data element
///
template<class T>
inline const T& dataAt(const T* pointer, size_t ix, size_t iy, size_t iz,
    size_t nx,
    size_t ny) {
    return *dataAtRawConst((const char*)pointer, ix * sizeof(T), iy * sizeof(T),
            iz * sizeof(T),
            nx, ny);
}

///
/// \brief dataAt gets the data at the given address
/// \param pointer the pointer to the data
/// \param ix the x index
/// \param iy the y index
/// \param iz the z index
/// \param nx the number of elements in x direction  (in bytes)
/// \param ny the number of elements in z direction  (in bytes)
/// \return the data element
///
template<class T>
inline  T& dataAt(T* pointer, size_t ix, size_t iy, size_t iz, size_t nx,
    size_t ny) {
    return *dataAtRaw((char*)pointer, ix * sizeof(T), iy * sizeof(T),
            iz * sizeof(T),
            nx, ny);
}

///
/// \brief calculateIndex calculates the index for the given coordinates
/// \param x the x index
/// \param y the y index
/// \param z the z index
/// \param nx the number of cells in x direction
/// \param ny the number of cells in y direction
/// \return the linear index
///
inline size_t calculateIndex(size_t x, size_t y, size_t z, size_t nx,
    size_t ny) {
    return z * nx * ny + y * nx + x;
}


