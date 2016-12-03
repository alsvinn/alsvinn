#pragma once
#include "alsfvm/equation/euler/ConservedVariables.hpp"
#include "alsfvm/equation/euler/AllVariables.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include <algorithm>

namespace alsfvm { namespace numflux { namespace euler { 

	///
	/// This is a utility class that only has the method computeFlux
	/// This will compute the HLL (Harten-van Leer-Lax) flux. 
	/// See eg.  http://link.springer.com/chapter/10.1007/978-3-662-03490-3_10#page-1 (requires springerlink).
	///
	/// This class is meant to be used with EulerNumericalFluxCPU or 
	/// EulerNumericalFluxGPU
	///
    template<int nsd>
    class HLL {
	public:

        ///
        /// \brief name is "hll"
        ///
        static const std::string name;

        typedef typename Types<nsd>::rvec rvec;
		
		///
		/// Computes the flux. Ie. computes
		/// \f[F(\mathrm{left}, \mathrm{right})\f]
		/// for the given direction.
		/// \todo Document this better
		///
		template<int direction>
        __device__ __host__ inline static real computeFlux(const equation::euler::Euler<nsd> eq, const equation::euler::AllVariables<nsd>& left,
			const equation::euler::AllVariables<nsd>& right, equation::euler::ConservedVariables<nsd>& F)
		{

			static_assert(direction < 3, "We only support three dimensions.");
			real speedLeft;
			real speedRight;

            computeHLLSpeeds<direction>(eq, left, right, speedLeft, speedRight);

			if (speedLeft == 0) {
                F = equation::euler::ConservedVariables<nsd>(0, rvec(0), 0);
			}
			else if (speedLeft > 0) {
                eq.template computePointFlux<direction>(left, F);
			}
			else if (speedRight < 0) {
                eq.template computePointFlux<direction>(right, F);
			}
			else {
                equation::euler::ConservedVariables<nsd> leftFlux, rightFlux;
                eq.template computePointFlux<direction>(left, leftFlux);
                eq.template computePointFlux<direction>(right, rightFlux);
				F = (speedRight*leftFlux - speedLeft*rightFlux + speedRight*speedLeft*(right.conserved() - left.conserved())) / (speedRight - speedLeft);
			}

			return fmax(fabs(speedLeft), fabs(speedRight));
		}

		/// 
		/// Computes the wave speeds for the given direction
		/// \param[in] left the values on the left side of the grid cell ("left" after we align the grid to the direction)
		/// \param[in] right the values on the right side of the grid cell
        /// \param[in] eq the equation instance
		/// \param[out] speedLeft the speed to the left side
		/// \param[out] speedRight the speed to the right
		/// \todo Document this better
		///
		template<int direction>
        __device__ __host__ inline static void computeHLLSpeeds(const equation::euler::Euler<nsd> eq, const equation::euler::AllVariables<nsd>& left,
			const equation::euler::AllVariables<nsd>& right, real& speedLeft, real& speedRight) {

            static_assert(direction < 3, "We only support up to three dimensions.");

            const real waveLeft = sqrt(left.rho);
            const real waveRight = sqrt(right.rho);

			const real rho = (left.rho + right.rho) / 2;
            const rvec u = (waveLeft * left.u + waveRight * right.u) / (waveLeft + waveRight);

			const real p = (left.p * waveLeft + right.p * waveRight) / (waveLeft + waveRight);

            const real cs = sqrt(eq.getGamma() * p / rho);

            speedLeft = fmin(left.u[direction] - sqrt(eq.getGamma() * left.p / left.rho), u[direction] - cs);
            speedRight = fmax(right.u[direction] + sqrt(eq.getGamma() * right.p / right.rho), u[direction] + cs);

		}
    };
} // namespace alsfvm
} // namespace numflux
} // namespace euler

