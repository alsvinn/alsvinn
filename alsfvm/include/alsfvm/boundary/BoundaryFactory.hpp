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
#include <string>
#include "alsfvm/boundary/Boundary.hpp"
#include "alsfvm/DeviceConfiguration.hpp"
namespace alsfvm {
namespace boundary {

class BoundaryFactory {
public:
    ///
    /// Instantiates the boundary factory
    /// \param name the name of the boundary type
    /// Parameter | Description
    /// ----------|------------
    /// "neumann"   | Neumann boundary conditions
    /// "periodic"  | Periodic boundary conditions
    ///
    /// \param deviceConfiguration the device configuration
    ///
    BoundaryFactory(const std::string& name,
        alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration);

    ///
    /// Creates the new boundary
    /// \param ghostCellSize the number of ghost cell to use on each side.
    ///
    alsfvm::shared_ptr<Boundary> createBoundary(size_t ghostCellSize);

private:
    std::string name;
    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration;
};
} // namespace alsfvm
} // namespace boundary
