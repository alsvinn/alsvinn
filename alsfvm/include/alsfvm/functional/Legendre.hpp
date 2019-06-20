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

//! Computes the spatial integral of the n-th Legendre polynomial,
//! ie it will compute
//!
//! \f[\int_D L_k(x)L_n(y)L_k(u(x, y))\; dx\; dy\f]
//!
//! where \f$L_n\f$ is the n-th Legendre polynomial.
//!
//! You have the option of scaling the input u to the interval [-1,1] by specifying the keywords
//! *maxValue* and *minValue*, this will essentially compute
//!
//! \f[\int_D L_k(x)L_n(y)L_m(\frac{u(x, y)-\mathrm{minValue}}{\mathrm{maxValue}-\mathrm{minValue}})\;dx\f]
//!
//! In terms of pairing between one point Young measures and test functions, this corresponds to
//!
//! \f[\langle \psi, \langle \nu, g\rangle \rangle\f]
//!
//! where \f$\psi(x,y)= L_k(x)L_n(y)\f$ and \f$g(\xi) = L_m(\xi)\f$.
//!
//! \note We always scale the polynomials in the spatial direction (ie. \f$L_k(x)L_n(y)\f$)
//! \note The computation of the Legendre polynomials are done through the boost library,
//!        see http://www.boost.org/doc/libs/1_46_1/libs/math/doc/sf_and_dist/html/math_toolkit/special/sf_poly/legendre.html
//!        for any implementation details. In short, \f$L_n(x)=boost::math::legendre_p(n,x)\f$
//!
class Legendre : public Functional {
public:
    //! The following parameters are accepted through parameters
    //!
    //!    Name      | Description
    //!    ----------|-------------
    //!    minValue  | minimum value that the solution can obtain
    //!    maxValue  | maximum value that the solution can obtain
    //!    degree_k  | the degree of the polynomial \f$L_k(x)\f$
    //!    degree_n  | the degree of the polynomial \f$L_n(x)\f$
    //!    degree_m  | the degree of the polynomial \f$L_m(u(x,y))\f$
    //!    variables | the variables to compute for (space separated)
    Legendre(const Parameters& parameters);

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
    //!
    //! @param[in] weight the current weight to be applied to the functional. Ie, the functional should compute
    //!                   \code{.cpp}
    //!                   conservedVolumeOut += weight + f(conservedVolumeIn)
    //!                   \endcode
    //! @param[in] grid the current grid
    //!
    virtual void operator()(volume::Volume& conservedVolumeOut,
        const volume::Volume& conservedVolumeIn,
        const real weight,
        const grid::Grid& grid
    ) override;

    //! Returns ivec3{1,1,1} -- we only need one element to represent this functional
    virtual ivec3 getFunctionalSize(const grid::Grid& grid) const override;


private:
    const real minValue = -1;
    const real maxValue = 1;
    const int degree_k = 1;
    const int degree_n = 1;
    const int degree_m = 1;

    std::vector<std::string> variables;
};
} // namespace functional
} // namespace alsfvm
