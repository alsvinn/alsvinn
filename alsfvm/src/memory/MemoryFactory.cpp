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

#include "alsfvm/memory/MemoryFactory.hpp"
#include "alsutils/error/Exception.hpp"
#include "alsfvm/memory/HostMemory.hpp"
#ifdef  ALSVINN_HAVE_CUDA
    #include "alsfvm/cuda/CudaMemory.hpp"
#endif
namespace alsfvm {
namespace memory {


///
/// \param deviceConfiguration the deviceConfiguration to use (this is mostly only relevant for GPU, on CPU it can be empty)
///
MemoryFactory::MemoryFactory(alsfvm::shared_ptr<DeviceConfiguration>&
    deviceConfiguration)
    : deviceConfiguration(deviceConfiguration) {
}

///
/// Creates scalar memory of the given size
/// \param nx the number of real numbers to store in X direction
/// \param ny the number of real numbers to store in Y direction
/// \param nz the number of real numbers to store in Z direction
/// \note nx, ny, nz is in number of reals
/// \returns the pointer to the newly created memory area.
///
alsfvm::shared_ptr<Memory<real> > MemoryFactory::createScalarMemory(size_t nx,
    size_t ny, size_t nz) {
    if (deviceConfiguration->getPlatform() == "cpu") {
        return alsfvm::shared_ptr<Memory<real> >(new HostMemory<real>(nx, ny, nz));
    } else if (deviceConfiguration->getPlatform() == "cuda") {
#ifdef ALSVINN_HAVE_CUDA
        return alsfvm::shared_ptr<Memory<real> >(new cuda::CudaMemory<real>(nx, ny,
                    nz));
#else
        THROW("CUDA is not enabled for this build");
#endif
    } else {
        THROW("Unknown memory type " << deviceConfiguration->getPlatform());
    }
}

const std::string& MemoryFactory::getPlatform() const {
    return deviceConfiguration->getPlatform();
}
}
}
