#pragma once
#include "alsfvm/volume/Volume.hpp"

namespace alsfvm {
	namespace numflux {

		///
		/// Base class for all numerical fluxes.
		///
		class NumericalFlux {
		public:
            virtual ~NumericalFlux() {}
			/// 
			/// Computes the numerical flux at each cell. 
			/// This will compute the net flux in the cell, ie.
			/// \f[
			/// \mathrm{output}_{i,j,k}=\frac{\Delta t}{\Delta x}\left(F(u_{i+1,j,k}, u_{i,j,k})-F(u_{i,j,k}, u_{i-1,j,k})\right)+
			///                         \frac{\Delta t}{\Delta y}\left(F(u_{i,j+1,k}, u_{i,j,k})-F(u_{i,j,k}, u_{i,j-1,k})\right)+
			///                         \frac{\Delta t}{\Delta z}\left((F(u_{i,j,k+1}, u_{i,j,k})-F(u_{i,j,k}, u_{i,j,k-1})\right)
			/// \f]
            /// \param[in] conservedVariables the conservedVariables to read from (eg. for Euler: \f$\rho,\; \vec{m},\; E\f$.
			/// \param[out] waveSpeed the maximum wave speed in each direction
			/// \param[in] computeWaveSpeed should we compute the wave speeds?
			/// \param[out] output the output to write to
			/// \note this will calculate the extra variables on the fly.
			/// 
			virtual void computeFlux(const volume::Volume& conservedVariables,
				rvec3& waveSpeed, bool computeWaveSpeed,
				volume::Volume& output
				) = 0;
			/// 
			/// \returns the number of ghost cells this specific flux requires
			///
			virtual size_t getNumberOfGhostCells() = 0;
		};
	}
} // namespace numflux
