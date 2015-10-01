#pragma once
#include "alsfvm/reconstruction/Reconstruction.hpp"

namespace alsfvm { namespace reconstruction { 

	template<size_t order>
    class WENOCUDA : public Reconstruction {
    public:
		///
		/// Performs reconstruction.
		/// \param[in] inputVariables the variables to reconstruct.
		/// \param[in] direction the direction:
		/// direction | explanation
		/// ----------|------------
		///     0     |   x-direction
		///     1     |   y-direction
		///     2     |   z-direction
		///
		/// \param[in] indicatorVariable the variable number to use for
		/// stencil selection. We will determine the stencil based on
		/// inputVariables->getScalarMemoryArea(indicatorVariable).
		///
		/// \param[out] leftOut at the end, will contain the left interpolated values
		///                     for all grid cells in the interior.
		///
		/// \param[out] rightOut at the end, will contain the right interpolated values
		///                     for all grid cells in the interior.
		///
		virtual void performReconstruction(const volume::Volume& inputVariables,
			size_t direction,
			size_t indicatorVariable,
			volume::Volume& leftOut,
			volume::Volume& rightOut);

		///
		/// \brief getNumberOfGhostCells returns the number of ghost cells we need
		///        for this computation
		/// \return order.
		///
		virtual size_t getNumberOfGhostCells() {
			return order;
		}
    };
} // namespace alsfvm
} // namespace reconstruction
