#pragma once
#include "alsfvm/numflux/NumericalFlux.hpp"
#include "alsfvm/grid/Grid.hpp"
#include "alsfvm/reconstruction/Reconstruction.hpp"

namespace alsfvm { namespace numflux { namespace euler { 

	///
	/// The class to compute numerical flux on the CPU
	/// The template argument Flux is used to choose the concrete flux
    /// The template argument dimension is to choose the correct dimension
    /// (1 up to and including 3 is supported).
	///
    template<class Flux, size_t dimension>
    class NumericalFluxCPU : public NumericalFlux {
    public:

        NumericalFluxCPU(const grid::Grid& grid,
                         std::shared_ptr<reconstruction::Reconstruction>& reconstruction,
                         std::shared_ptr<DeviceConfiguration>& deviceConfiguration
                         );

        ///
        /// \brief computeFlux
        /// \param conservedVariables
        /// \param cellLengths
        /// \param output
        /// \todo This needs to fix the way it computes the flux (adds another minus that must be fixed)
        ///       (It computes correctly, it is maybe just a bit unclear what it does)
        ///
        virtual void computeFlux(const volume::Volume& conservedVariables,
			const rvec3& cellLengths,
			volume::Volume& output
			);

		/// 
		/// \returns the number of ghost cells this specific flux requires
		///
		virtual size_t getNumberOfGhostCells();

    private:
        std::shared_ptr<reconstruction::Reconstruction> reconstruction;
        std::shared_ptr<volume::Volume> left;
        std::shared_ptr<volume::Volume> right;
    };

} // namespace alsfvm
} // namespace numflux
} // namespace euler

