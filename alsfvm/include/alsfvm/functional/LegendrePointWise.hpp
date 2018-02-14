#pragma once
#include "alsfvm/functional/Functional.hpp"

namespace alsfvm {
namespace functional {

//! Computes the spatial integral of the n-th Legendre polynomial,
//! ie it will output \f$v(x,y)\f$, where
//!
//! \f[v(x,y) = L_k(u(x, y))\f]
//!
//! where \f$L_n\f$ is the n-th Legendre polynomial.
//!
//! You have the option of scaling the input u to the interval [-1,1] by specifying the keywords
//! *maxValue* and *minValue*, this will essentially compute
//!
//! \f[v(x,y) = L_m(\frac{u(x, y)-\mathrm{minValue}}{\mathrm{maxValue}-\mathrm{minValue}})\f]
//!
//! In terms of pairing between one point Young measures and test functions, this corresponds to
//!
//! \f[\langle \nu, g\rangle \f]
//!
//! where  and \f$g(\xi) = L_m(\xi)\f$.
//!
//! \note The computation of the Legendre polynomials are done through the boost library,
//!        see http://www.boost.org/doc/libs/1_46_1/libs/math/doc/sf_and_dist/html/math_toolkit/special/sf_poly/legendre.html
//!        for any implementation details. In short, \f$L_n(x)=boost::math::legendre_p(n,x)\f$
//!
class LegendrePointWise : public Functional {
    public:
        //! The following parameters are accepted through parameters
        //!
        //!    Name      | Description
        //!    ----------|-------------
        //!    minValue  | minimum value that the solution can obtain
        //!    maxValue  | maximum value that the solution can obtain
        //!    degree    | the degree of the polynomial \f$L_n(x)\f$
        //!    variables | the variables to compute for (space separated)
        LegendrePointWise(const Parameters& parameters);

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
        //! @param[in] extraVolume the state of the extra volume
        //!
        //! @param[in] weight the current weight to be applied to the functional. Ie, the functional should compute
        //!                   \code{.cpp}
        //!                   conservedVolumeOut += weight + f(conservedVolumeIn)
        //!                   \endcode
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
        const real minValue = -1;
        const real maxValue = 1;
        const int degree = 1;

        std::vector<std::string> variables;
};
} // namespace functional
} // namespace alsfvm
