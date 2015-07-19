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
    class HLL {
	public:
		
		///
		/// Computes the flux. Ie. computes
		/// \f[F(\mathrm{left}, \mathrm{right})\f]
		/// for the given direction.
		/// \todo Document this better
		///
		template<int direction>
		inline static void computeFlux(const equation::euler::AllVariables& left,
			const equation::euler::AllVariables& right, equation::euler::ConservedVariables& F)
		{

			static_assert(direction < 3, "We only support three dimensions.");
			real speedLeft;
			real speedRight;

			computeHLLSpeeds<direction>(left, right, speedLeft, speedRight);

			if (speedLeft == 0) {
				F = equation::euler::ConservedVariables(0, 0, 0, 0, 0);
			}
			else if (speedLeft > 0) {
				equation::euler::Euler::computePointFlux<direction>(left, F);
			}
			else if (speedRight < 0) {
				equation::euler::Euler::computePointFlux<direction>(right, F);
			}
			else {
				ConservedVariables leftFlux, rightFlux;

				equation::euler::Euler::computePointFlux<direction>(left, leftFlux);
				equation::euler::Euler::computePointFlux<direction>(right, rightFlux);
				F = (speedRight*leftFlux - speedLeft*rightFlux + speedRight*speedLeft*(right.conserved() - left.conserved())) / (speedRight - speedLeft);
			}
		}

		/// 
		/// Computes the wave speeds for the given direction
		/// \param[in] left the values on the left side of the grid cell ("left" after we align the grid to the direction)
		/// \param[in] right the values on the right side of the grid cell
		/// \param[out] speedLeft the speed to the left side
		/// \param[out] speedRight the speed to the right
		/// \todo Document this better
		///
		template<int direction>
		inline static void computeHLLSpeeds(const equation::euler::AllVariables& left,
			const equation::euler::AllVariables& right, real& speedLeft, real& speedRight) {

			static_assert(direction < 3, "We only support three dimensions.");

			const real waveLeft = sqrt(left.rho);
			const real waveRight = sqrt(right.rho);

			const real rho = (left.rho + right.rho) / 2;
			const rvec3 u = (waveLeft * left.u + waveRight * right.u) / (waveLeft + waveRight);
			const real p = (left.p * waveLeft + right.p * waveRight) / (waveLeft + waveRight);

			const real cs = sqrt(GAMMA * p / rho);

			speedLeft = std::min(left.u[direction] - sqrt(GAMMA * left.p / left.rho), u[direction] - cs);
			speedRight = std::max(right.u[direction] + sqrt(GAMMA * right.p / right.rho), u[direction] + cs);

		}
    };
} // namespace alsfvm
} // namespace numflux
} // namespace euler

