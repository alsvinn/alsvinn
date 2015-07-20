#pragma once
#include "alsfvm/numflux/NumericalFlux.hpp"

namespace alsfvm { namespace numflux { namespace euler { 

	///
	/// The class to compute numerical flux on the CPU
	/// The template argument Flux is used to choose the concrete flux
	///
	template<class Flux>
    class NumericalFluxCPU : public NumericalFlux {
    public:


		virtual void computeFlux(const volume::Volume& conservedVariables,
			const volume::Volume& extraVariables,
			const rvec3& cellLengths,
			volume::Volume& output
			);

		/// 
		/// \returns the number of ghost cells this specific flux requires
		///
		virtual size_t getNumberOfGhostCells();

    };

} // namespace alsfvm
} // namespace numflux
} // namespace euler
