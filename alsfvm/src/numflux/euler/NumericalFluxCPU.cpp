#include "alsfvm/numflux/euler/NumericalFluxCPU.hpp"
#include "alsfvm/numflux/euler/HLL.hpp"

namespace alsfvm { namespace numflux { namespace euler { 

	template<class Flux>
	virtual void NumericalFluxCPU<Flux>::computeFlux(const volume::Volume& conservedVariables,
		const volume::Volume& extraVariables,
		const rvec3& cellLengths,
		volume::Volume& output
		) 
	{

		const int nx = conservedVariables.getNumberOfXCells();
		const int ny = conservedVariables.getNumberOfYCells();
		const int nz = conservedVariables.getNumberOfZCells();

		for (size_t k = 1; k < nz - 1; k++) {
			for (size_t j = 1; j < ny - 1; j++) {
				for (size_t i = 1; i < nx - 1; i++) {
					equation::euler::ConservedVariables flux;



				}
			}
		}
	}

	/// 
	/// \returns the number of ghost cells this specific flux requires
	///
	template<class Flux>
	virtual size_t getNumberOfGhostCells() {
		return 1;
	}

}
}
}
