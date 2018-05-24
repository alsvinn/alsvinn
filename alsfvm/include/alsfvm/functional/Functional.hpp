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
#include "alsutils/parameters/Parameters.hpp"
#include "alsfvm/grid/Grid.hpp"

namespace alsfvm {
namespace functional {


//! @brief Abstract base class to represent a functional
//!
//! A functional is an abstract type to represent a map from the solution
//! to some other space.
class Functional {
public:
    //! To be used to pass parameters to the constructors
    typedef alsutils::parameters::Parameters Parameters;

    virtual ~Functional() {}

    //! Computes the operator value on the givne input data
    //!
    //! @note In order to support time integration, the result should be
    //!       added to conservedVolumeOut and extraVolumeOut, not overriding
    //!       it.
    //!
    //! @param[out] conservedVolumeOut at the end, should have the contribution
    //!             of the functional for the conservedVariables
    //!
    //! @param[out] extraVolumeOut at the end, should have the contribution
    //!             of the functional for the extraVariables
    //!
    //! @param[in] conservedVolumeIn the state of the conserved variables
    //!
    //! @param[in] extraVolumeIn the state of the extra volume
    //!
    //! @param[in] weight the current weight to be applied to the functional. Ie, the functional should compute
    //!                   \code{.cpp}
    //!                   conservedVolumeOut += weight + f(conservedVolumeIn)
    //!                   \endcode
    //! @param[in] grid the grid to work on
    //!
    virtual void operator()(volume::Volume& conservedVolumeOut,
        volume::Volume& extraVolumeOut,
        const volume::Volume& conservedVolumeIn,
        const volume::Volume& extraVolumeIn,
        const real weight,
        const grid::Grid& grid
    ) = 0;

    //! Returns the number of elements needed to represent the functional
    //!
    //! Eg. returning ivec3{1,1,1} would imply that operator() should be
    //!     called with conservedVolumeOut and extraVolumeOut being
    //!     of size {1,1,1}.
    virtual ivec3 getFunctionalSize(const grid::Grid& grid) const = 0;
};

typedef alsfvm::shared_ptr<Functional> FunctionalPointer;
} // namespace functional
} // namespace alsfvm
