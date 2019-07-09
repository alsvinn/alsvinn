/* Copyright (c) 2018, 2019 ETH Zurich, Kjetil Olsen Lye
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
#include <thrust/device_vector.h>

namespace alsfvm {
namespace functional {

//! This is basically the functional version of
//! stats/StructureCube.
//!
//! @todo Refactor this to avoid code duplication.
class StructureCubeCUDA : public Functional {
public:
    StructureCubeCUDA(const Functional::Parameters& parameters);

    //! Computes the operator value on the givne input data
    //!
    //! @note In order to support time integration, the result should be
    //!       added to conservedVolumeOut and extraVolumeOut, not overriding
    //!       it.
    //!
    //! @param[out] conservedVolumeOut at the end, should have the contribution
    //!             of the functional for the conservedVariables
    //!
    //! @param[in] conservedVolumeIn the state of the conserved variables
    //!
    //! @param[in] weight the current weight to be applied to the functional. Ie, the functional should compute
    //!                   \code{.cpp}
    //!                   conservedVolumeOut += weight + f(conservedVolumeIn)
    //!                   \endcode
    //!
    //! @param[in] grid the grid to use
    //!
    virtual void operator()(volume::Volume& conservedVolumeOut,
        const volume::Volume& conservedVolumeIn,
        const real weight,
        const grid::Grid& grid
    ) override;

    //! Returns the number of elements needed to represent the functional
    //!
    //! Returns {numberOfH, 1,1}
    virtual ivec3 getFunctionalSize(const grid::Grid& grid) const override;

private:

    const real p;
    const int numberOfH;


    // For now we use thurst's version of reduce, and to make everything
    // play nice, we use thrust device vector to hold the temporary results.
    thrust::device_vector<real> structureOutput;
};
} // namespace functional
} // namespace alsfvm
