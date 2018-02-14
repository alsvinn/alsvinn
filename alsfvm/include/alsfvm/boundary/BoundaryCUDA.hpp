#pragma once
#include "alsfvm/boundary/Boundary.hpp"
namespace alsfvm {
namespace boundary {

template<class BoundaryConditions>
class BoundaryCUDA : public Boundary {
    public:

        ///
        /// Constructs a new instance
        /// \param numberOfGhostCells the number of ghost cells on each side to use.
        ///
        BoundaryCUDA(size_t numberOfGhostCells);

        ///
        /// Applies the neumann boundary to the volumes supplied.
        /// For a ghost size of 1, we set
        /// \f[U_0 = U_1\qquad\mathrm{and}\qquad U_N=U_{N-1}\f]
        ///
        ///
        /// Applies the boundary conditions to the volumes supplied.
        /// \param volume the volume to apply the boundary condition to
        /// \param grid the active grid
        /// \todo Better handling of corners.
        ///
        virtual void applyBoundaryConditions(volume::Volume& volume,
            const grid::Grid& grid);

    private:
        size_t numberOfGhostCells;
};
} // namespace alsfvm
} // namespace boundary
