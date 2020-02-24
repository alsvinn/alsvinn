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
//! Computes the thermodynamic entropy, see
//!
//! Fjordholm, U. S., Mishra, S., & Tadmor, E. (2012). Arbitrarily high-order accurate entropy stable essentially nonoscillatory schemes for systems of conservation laws, 50(2), 544–573.
//!
//!  for a definition (2.16), E.
//!
//!  Note that we compute the spatial integral of the entropy at all points.
//!
//! Specifically, we set
//!
//! \f[ s = \log(p)-\gamma\log(p)\f]
//!
//! where \f$\gamma>0\f$ is the gas constant. We set the entropy $E$ to be
//!
//! \f[E=\frac{-\rho s}{\gamma-1}\f]
//!
//! This functional computes
//!
//! \f[\int_D E \;dx\f]
//!
//!
//!
//!
class LogEntropy : public Functional {
public:

    //! Uses no parameter
    LogEntropy(const Parameters& parameters);

    //! Computes the operator value on the given input data
    //!
    //! @note In order to support time integration, the result should be
    //!       added to conservedVolumeOut and extraVolumeOut, not overriding
    //!       it.
    //!
    //! @param[out] conservedVolumeOut at the end, should have the contribution
    //!             of the functional for the conservedVariables
    //!
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

    //! Returns grid.getDimensions()
    virtual ivec3 getFunctionalSize(const grid::Grid& grid) const override;


private:

    const real gamma;

};
} // namespace functional
} // namespace alsfvm
