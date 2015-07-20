#pragma once
#include "alsfvm/volume/Volume.hpp"

namespace alsfvm {
	namespace numflux {

		///
		/// Base class for all numerical fluxes.
		///
		class NumericalFlux {
		public:

			/// 
			/// Computes the numerical flux at each cell. 
			/// This will compute the net flux in the cell, ie.
			/// \f[
			/// \mathrm{output}_{i,j,k}=\frac{\Delta t}{\Delta x}\left(F(u_{i+1,j,k}, u_{i,j,k})-F(u_{i,j,k}, u_{i-1,j,k})\right)+
			///                         \frac{\Delta t}{\Delta y}\left(F(u_{i,j+1,k}, u_{i,j,k})-F(u_{i,j,k}, u_{i,j-1,k})\right)+
			///                         \frac{\Delta t}{\Delta z}\left((F(u_{i,j,k+1}, u_{i,j,k})-F(u_{i,j,k}, u_{i,j,k-1})\right)
			/// \f]
			/// \param[in] conservedVariables the conservedVariables to read from (eg. for Euler: \f$\rho,\; \vec{m},\; E\f$)
			/// \param[in] extraVariables some extra variables that the equation needs (eg. for Euler: \f$p, \vec{u}\f$)
			/// \param[in] cellLengths contains the cell length in each direction. So
			///            \f{eqnarray*}{
			///             \Delta x = \mathrm{cellLengths.x}\\
			///             \Delta y = \mathrm{cellLengths.y}\\
			///             \Delta z = \mathrm{cellLengths.z}\\
			///            \f}
			/// \param[out] output the output to write to
			/// 
			virtual void computeFlux(const volume::Volume& conservedVariables,
				const volume::Volume& extraVariables,
				const rvec3& cellLengths,
				volume::Volume& output
				) = 0;

			/// 
			/// \returns the number of ghost cells this specific flux requires
			///
			virtual size_t getNumberOfGhostCells() = 0;
		};
	}
} // namespace numflux
