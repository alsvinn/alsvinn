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
#include "alsfvm/boundary/Boundary.hpp"
namespace alsfvm {
namespace boundary {

template<class BoundaryConditions>
class BoundaryCPU : public Boundary {
public:

    ///
    /// Constructs a new instance
    /// \param numberOfGhostCells the number of ghost cells on each side to use.
    ///
    BoundaryCPU(size_t numberOfGhostCells);

    ///
    /// Applies the neumann boundary to the volumes supplied.
    /// For a ghost size of 1, we set
    /// \f[U_0 = U_1\qquad\mathrm{and}\qquad U_N=U_{N-1}\f]
    ///
    ///
    /// Applies the boundary conditions to the volumes supplied.
    /// \param volume the volume to apply the boundary condition to
    /// \param grid the active grid
    /// \todo Better handling of corners.
    ///
    virtual void applyBoundaryConditions(volume::Volume& volume,
        const grid::Grid& grid);

private:
    size_t numberOfGhostCells;
};
} // namespace alsfvm
} // namespace boundary
