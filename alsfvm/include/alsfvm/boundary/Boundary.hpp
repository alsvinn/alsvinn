#pragma once
#include "alsfvm/volume/Volume.hpp"
#include "alsfvm/grid/Grid.hpp"

namespace alsfvm { namespace boundary { 

    class Boundary {
    public:
		///
		/// Applies the boundary conditions to the volumes supplied.
		/// \param volume the volume to apply the boundary condition to
		///
		virtual void applyBoundaryConditions(volume::Volume& volume,
			const grid::Grid& grid) = 0;

    };
} // namespace alsfvm
} // namespace boundary
