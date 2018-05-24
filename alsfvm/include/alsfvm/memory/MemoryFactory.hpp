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
#include "alsfvm/memory/MemoryBase.hpp"
#include "alsfvm/memory/Memory.hpp"
#include "alsfvm/DeviceConfiguration.hpp"
#include "alsfvm/types.hpp"
#include <functional>
#include <map>

namespace alsfvm {
namespace memory {
class MemoryFactory {
public:
    ///
    /// The base memory pointer
    ///
    typedef alsfvm::shared_ptr < MemoryBase > MemoryPtr;


    ///
    /// \param deviceConfiguration the deviceConfiguration to use (this is mostly only relevant for GPU, on CPU it can be empty)
    ///
    MemoryFactory(alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration);

    ///
    /// Creates scalar memory of the given size
    /// \param nx the number of real numbers to store in X direction
    /// \param ny the number of real numbers to store in Y direction
    /// \param nz the number of real numbers to store in Z direction
    /// \note nx, ny, nz is in number of reals
    /// \returns the pointer to the newly created memory area.
    ///
    alsfvm::shared_ptr<Memory<real> > createScalarMemory(size_t nx, size_t ny,
        size_t nz);

    const std::string& getPlatform() const;
private:

    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration;
};

typedef alsfvm::shared_ptr<MemoryFactory> MemoryFactoryPointer;
}
}
