#pragma once
#include "alsfvm/numflux/NumericalFlux.hpp"
#include "alsfvm/DeviceConfiguration.hpp"
#include "alsfvm/reconstruction/Reconstruction.hpp"
#include "alsfvm/grid/Grid.hpp"
namespace alsfvm { namespace numflux { 

	template<class Flux, class Equation, size_t dimension>
    class NumericalFluxCUDA : public NumericalFlux  {
    public:
		NumericalFluxCUDA(const grid::Grid& grid,
			alsfvm::shared_ptr<reconstruction::Reconstruction>& reconstruction,
			alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration
			);

		/// 
		/// Computes the numerical flux at each cell. 
		/// This will compute the net flux in the cell, ie.
		/// \f[
		/// \mathrm{output}_{i,j,k}=\frac{\Delta t}{\Delta x}\left(F(u_{i+1,j,k}, u_{i,j,k})-F(u_{i,j,k}, u_{i-1,j,k})\right)+
		///                         \frac{\Delta t}{\Delta y}\left(F(u_{i,j+1,k}, u_{i,j,k})-F(u_{i,j,k}, u_{i,j-1,k})\right)+
		///                         \frac{\Delta t}{\Delta z}\left((F(u_{i,j,k+1}, u_{i,j,k})-F(u_{i,j,k}, u_{i,j,k-1})\right)
		/// \f]
		/// \param[in] conservedVariables the conservedVariables to read from (eg. for Euler: \f$\rho,\; \vec{m},\; E\f$)
		/// \param[in] cellScaling contains the cell length in each direction. So
		///            \f{eqnarray*}{
		///             \Delta t/\Delta x = \mathrm{cellLengths.x}\\
		            ///             \Delta t/\Delta y = \mathrm{cellLengths.y}\\
		            ///             \Delta t/\Delta z = \mathrm{cellLengths.z}\\
					///            \f}
		/// \param[out] output the output to write to
		///
		/// \note this will calculate the extra variables on the fly.
		/// 
		virtual void computeFlux(const volume::Volume& conservedVariables,
			const rvec3& cellScaling,
			volume::Volume& output
			);

		/// 
		/// \returns the number of ghost cells this specific flux requires
		///
		virtual size_t getNumberOfGhostCells();
	private:
		alsfvm::shared_ptr<reconstruction::Reconstruction> reconstruction;
		alsfvm::shared_ptr<volume::Volume> left;
		alsfvm::shared_ptr<volume::Volume> right;
		alsfvm::shared_ptr<volume::Volume> fluxOutput;
    };
} // namespace alsfvm
} // namespace numflux
