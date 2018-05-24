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
#include "alsfvm/functional/Functional.hpp"

namespace alsfvm {
namespace functional {

//!
//! This just dumps the solution at the current time
//!
class Identity : public Functional {
public:

    //! Uses no parameter
    Identity(const Parameters& parameters);

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
    //!
    //! @param[in] grid the grid to use
    //!
    virtual void operator()(volume::Volume& conservedVolumeOut,
        volume::Volume& extraVolumeOut,
        const volume::Volume& conservedVolumeIn,
        const volume::Volume& extraVolumeIn,
        const real weight,
        const grid::Grid& grid
    ) override;

    //! Returns grid.getDimensions()
    virtual ivec3 getFunctionalSize(const grid::Grid& grid) const override;


private:

};
} // namespace functional
} // namespace alsfvm
