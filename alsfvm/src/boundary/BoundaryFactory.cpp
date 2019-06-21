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

#include "alsutils/config.hpp"
#include "alsfvm/boundary/BoundaryFactory.hpp"
#include "alsutils/error/Exception.hpp"
#include "alsfvm/boundary/BoundaryCPU.hpp"
#include "alsfvm/boundary/Neumann.hpp"
#include "alsfvm/boundary/Periodic.hpp"
#ifdef ALSVINN_HAVE_CUDA
    #include "alsfvm/boundary/BoundaryCUDA.hpp"
#endif

namespace alsfvm {
namespace boundary {
///
/// Instantiates the boundary factory
/// \param name the name of the boundary type
///   Parameter | Description
///   ----------|------------
/// "neumann"   | Neumann boundary conditions
/// "periodic"  | Periodic boundary conditions
///
/// \param deviceConfiguration the device configuration
///
BoundaryFactory::BoundaryFactory(const std::string& name,
    alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration)
    : name(name), deviceConfiguration(deviceConfiguration) {

}

///
/// Creates the new boundary
/// \param ghostCellSize the number of ghost cell to use on each side.
///
alsfvm::shared_ptr<Boundary> BoundaryFactory::createBoundary(
    size_t ghostCellSize) {
    if (deviceConfiguration->getPlatform() == "cpu") {
        if (name == "neumann") {
            return alsfvm::shared_ptr<Boundary>(new BoundaryCPU<Neumann>(ghostCellSize));
        } else if (name == "periodic") {
            return alsfvm::shared_ptr<Boundary>(new BoundaryCPU<Periodic>(ghostCellSize));
        } else {
            THROW("Unknown boundary type " << name);
        }
    }

#ifdef ALSVINN_HAVE_CUDA
    else if (deviceConfiguration->getPlatform() == "cuda") {
        if (name == "neumann") {
            return alsfvm::shared_ptr<Boundary>(new BoundaryCUDA<Neumann>(ghostCellSize));
        } else if (name == "periodic") {
            return alsfvm::shared_ptr<Boundary>(new BoundaryCUDA<Periodic>(ghostCellSize));
        } else {
            THROW("Unknown boundary type " << name);
        }
    }

#endif
    else {
        THROW("Unknown platform " << deviceConfiguration->getPlatform());
    }
}
}
}
