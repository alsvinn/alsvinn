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
#include "alsfvm/grid/Grid.hpp"
#include "alsfvm/mpi/domain/DomainInformation.hpp"
#include "alsfvm/mpi/Configuration.hpp"

namespace alsfvm {
namespace mpi {
namespace domain {
//! Abstract base class to do domain decomposition
class DomainDecomposition {
public:
    virtual ~DomainDecomposition() {}


    //! Decomposes the grid. The returned object is the local information
    //! for this node.
    //!
    //! @param configuration the configuration
    //! @param grid the grid
    virtual DomainInformationPtr decompose(ConfigurationPtr configuration,
        const grid::Grid& grid) = 0;
};
} // namespace domain
} // namespace mpi
} // namespace alsfvm
