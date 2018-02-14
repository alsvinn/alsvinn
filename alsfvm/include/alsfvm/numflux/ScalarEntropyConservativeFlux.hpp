#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/numflux/burgers/Godunov.hpp"
namespace alsfvm {
namespace numflux {

//! Implements the scalar entropy conservative flux
//! \f[F(u_i, u_{i+1}) = \left\{\begin{array}{lr}\frac{\psi_{i+1}-\psi_i}{v_{i+1}-v_i} & u_i\neq u_{i+1}\\ f(u_i) & \mathrm{otherwise}\end{array}\right.\f]
//!
//! Here \f$\psi\f$ is the entropy potential and \f$v\f$ is the entropy variables.
//!
//! See eg. http://www.cscamm.umd.edu/people/faculty/tadmor/pub/TV+entropy/Fjordholm_Mishra_Tadmor_SINUM2012.pdf
//! (Fjordholm et al, Arbitrarily high-order accurate entropy stable essentially nonoscillatory schemes for systems of conservation laws)
//!
template<class Equation>
class ScalarEntropyConservativeFlux {
public:
    ///
    /// \brief name is "tecno1"
    ///
    static const std::string name;

    template<int direction>
    __device__ __host__ inline static real computeFlux(const Equation& eq,
        const typename Equation::AllVariables& left,
        const typename Equation::AllVariables& right,
        typename Equation::ConservedVariables& F) {
        auto leftEntropyVariable = eq.computeEntropyVariables(left);
        auto rightEntropyVariable = eq.computeEntropyVariables(right);

        if (left.u ==
            right.u) {// || (leftEntropyVariable - rightEntropyVariable).norm() < 1e-6) {
            eq.template computePointFlux<direction>(left, F);
        } else {
            auto leftEntropyPotential = eq.computeEntropyPotential(left);
            auto rightEntropyPotential = eq.computeEntropyPotential(right);

            F.u = ((rightEntropyPotential - leftEntropyPotential) /
                    (rightEntropyVariable - leftEntropyVariable));
        }

        return fmax(eq.template computeWaveSpeed<direction>(left, left),
                eq.template computeWaveSpeed<direction>(right, right));
    }
};
} // namespace numflux
} // namespace alsfvm
