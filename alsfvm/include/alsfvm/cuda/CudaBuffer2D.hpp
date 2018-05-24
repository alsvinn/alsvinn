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
#include "alsfvm/cuda/CudaBuffer.hpp"
namespace alsfvm {
namespace cuda {

///
/// 1D cuda buffer
///
template<typename T>
class CudaBuffer2D : public CudaBuffer < T > {
public:
    ///
    /// \param nx the number of Ts in x direction
    /// \param ny the number of Ts in y direction
    ///
    CudaBuffer2D(size_t nx, size_t ny);
    virtual ~CudaBuffer2D();


    ///
    /// \returns the pointer to the data
    ///
    virtual T* getPointer();

    ///
    /// \returns the pointer to the data
    ///
    virtual const T* getPointer() const;


private:
    T* memoryPointer;
};
}
}