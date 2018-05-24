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
#include "alsfvm/volume/Volume.hpp"
#include "alsfvm/grid/Grid.hpp"

namespace alsfvm {
namespace boundary {

class Boundary {
public:
    ///
    /// Applies the boundary conditions to the volumes supplied.
    /// \param volume the volume to apply the boundary condition to
    /// \param grid the active grid
    ///
    virtual void applyBoundaryConditions(volume::Volume& volume,
        const grid::Grid& grid) = 0;

    //! Since we inherit, we have an empty virtual constructor
    virtual ~Boundary() {}

};
} // namespace alsfvm
} // namespace boundary
