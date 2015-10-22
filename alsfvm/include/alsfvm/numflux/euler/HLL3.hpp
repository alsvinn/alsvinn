#pragma once
#include "alsfvm/equation/euler/ConservedVariables.hpp"
#include "alsfvm/equation/euler/AllVariables.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include <algorithm>
#include <numeric>
#include "alsfvm/numflux/euler/HLL.hpp"

namespace alsfvm { namespace numflux { namespace euler {

///
/// This is a utility class that only has the method computeFlux
/// This will compute the HLL3 (Harten-van Leer-Lax 3 wave) flux.
/// See eg.  http://link.springer.com/chapter/10.1007/978-3-662-03490-3_10#page-1 (requires springerlink).
///
/// This class is meant to be used with EulerNumericalFluxCPU or
/// EulerNumericalFluxGPU
///
class HLL3 {
public:

    ///
    /// Computes the flux. Ie. computes
    /// \f[F(\mathrm{left}, \mathrm{right})\f]
    /// for the given direction.
    /// \todo Document this better
    ///
    template<int direction>
    __device__ __host__ inline static real computeFlux(const equation::euler::Euler& eq, const equation::euler::AllVariables& left,
                                   const equation::euler::AllVariables& right, equation::euler::ConservedVariables& F)
    {

        static_assert(direction < 3, "We only support three dimensions.");
        real speedLeft;
        real speedRight;
        real cs;
        equation::euler::ConservedVariables fluxLeft, fluxRight;

        computeHLLSpeeds<direction>(eq, left, right, speedLeft, speedRight, cs);
        eq.computePointFlux<direction>(left, fluxLeft);
        eq.computePointFlux<direction>(right, fluxRight);

        const real pressureLeft = left.p;
        const real pressureRight = right.p;

        const real udl = left.u[direction]-speedLeft;
        const real udr = right.u[direction]-speedRight;

        real aa = udr*right.rho - udl*left.rho;
        real sm = (right.m[direction]*udr - left.m[direction]*udl + pressureRight - pressureLeft) / aa;

        rvec3 us = ( fluxRight.m - fluxLeft.m - speedRight*right.m + speedLeft*left.m ) / aa;
        us[direction] = sm;

        if (speedLeft == 0) {
            F = equation::euler::ConservedVariables(0, 0, 0, 0, 0);
        }
        else if (speedLeft > 0) {
            F = fluxLeft;
        }
        else if (speedRight < 0) {
            F = fluxRight;
        }
        else if (sm >= 0) {
            equation::euler::ConservedVariables middle;
            middle.rho = left.rho*udl/(sm-speedLeft);
            middle.m = middle.rho * us;
            const real p = pressureLeft + left.rho*(left.u[direction]-sm)*udl;
            middle.E = (udl*left.E + pressureLeft*left.u[direction] - p*sm) / (sm-speedLeft);
            F = fluxLeft + speedLeft * (middle-left.conserved());
        } else {
            equation::euler::ConservedVariables middle;
            middle.rho = right.rho*udr/(sm-speedRight);
            middle.m = middle.rho * us;
            real p = pressureRight + right.rho*(right.u[direction]-sm)*udr;
            middle.E = (udr*right.E + pressureRight*right.u[direction] - p*sm) / (sm-speedRight);
            F = fluxRight + speedRight*(middle-right.conserved());
        }

		return fmax(fabs(speedLeft), fabs(speedRight));
    }

    ///
    /// Computes the wave speeds for the given direction
    /// \param[in] left the values on the left side of the grid cell ("left" after we align the grid to the direction)
    /// \param[in] right the values on the right side of the grid cell
    /// \param[out] speedLeft the speed to the left side
    /// \param[out] speedRight the speed to the right
    /// \param[out] cs the speed of sound
    /// \todo Document this better
    ///
    template<int direction>
    __device__ __host__ inline static void computeHLLSpeeds(const equation::euler::Euler& eq,
                                                            const equation::euler::AllVariables& left,
                                        const equation::euler::AllVariables& right, real& speedLeft, real& speedRight, real& cs) {

        const real waveLeft = sqrt(left.rho);
        const real waveRight = sqrt(right.rho);

        const real rho = (left.rho + right.rho)/2;
        const rvec3 u = (waveLeft * left.u + waveRight * right.u)/(waveLeft+waveRight);
        const real p = (waveLeft * left.p + waveRight * right.p)/(waveLeft+waveRight);

        // calculate extended fast speed cf at the left boundary
        const real cfLeft = sqrt( eq.getGamma()*left.p/left.rho );

        // calculate extended fast speed cf at the right boundary
        const real cfRight = sqrt( eq.getGamma()*right.p/right.rho );

        const real correct = 0.5*fmax(real(0),left.u[direction]-right.u[direction]);

        cs = sqrt(eq.getGamma() * p / rho);
        speedLeft = fmin(left.u[direction] + correct - cfLeft, u[direction] - cs);
        speedRight = fmax(right.u[direction] - correct + cfRight, u[direction] + cs);

    }
};
} // namespace alsfvm
} // namespace numflux
} // namespace euler

