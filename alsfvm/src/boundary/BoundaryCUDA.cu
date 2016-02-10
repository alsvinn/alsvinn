#include "alsfvm/boundary/BoundaryCUDA.hpp"
#include "alsfvm/boundary/Neumann.hpp"
#include "alsfvm/boundary/Periodic.hpp"
#include "cuda.h"


namespace alsfvm { namespace boundary { 

	namespace {
		template<class BoundaryConditions, bool top, bool xDir, bool yDir, bool zDir>
		__global__ void applyBoundaryConditionsDevice(memory::View<real> memoryArea, 
			const size_t numberOfXCells, const size_t numberOfYCells, const size_t numberOfZCells, 
			const size_t internalNumberOfXCells, const size_t internalNumberOfYCells, const size_t internalNumberOfZCells, const size_t numberOfGhostCells) {
			
			const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
			
			if (index > numberOfXCells * numberOfYCells * numberOfZCells) {
				return;
			}

			// We have
			// index = z * nx * ny + y * nx + x;
			const size_t xInternalFormat = xDir ? 0 : index % numberOfXCells;
			const size_t yInternalFormat = yDir ? 0 : (index / numberOfXCells) % numberOfYCells;
			const size_t zInternalFormat = zDir ? 0 : min(int((index) / (numberOfXCells * numberOfYCells)),
			int(numberOfZCells - 1));
			

			const size_t x = xInternalFormat + xDir * (numberOfGhostCells + internalNumberOfXCells * top);
			const size_t y = yInternalFormat + yDir * (numberOfGhostCells + internalNumberOfYCells * top);
			const size_t z = zInternalFormat + zDir * (numberOfGhostCells + internalNumberOfZCells * top);
			for (size_t ghostCell = 1; ghostCell <= numberOfGhostCells; ++ghostCell) {
				BoundaryConditions::applyBoundary(memoryArea, x, y, z, ghostCell, numberOfGhostCells, top, xDir, yDir, zDir);
			}
		}

		template<class BoundaryConditions, bool top, bool xDir, bool yDir, bool zDir>
		void applyBoundaryConditions(memory::View<real>& memoryArea, const size_t numberOfGhostCells) {
			const size_t numberOfXCells = xDir ? 1 : memoryArea.getNumberOfXCells();
			const size_t numberOfYCells = yDir ? 1 : memoryArea.getNumberOfYCells();
			const size_t numberOfZCells = zDir ? 1 : memoryArea.getNumberOfZCells();

			const size_t internalNumberOfXCells = memoryArea.getNumberOfXCells() - 2 * numberOfGhostCells - 1;
			const size_t internalNumberOfYCells = memoryArea.getNumberOfYCells() - 2 * numberOfGhostCells - 1;
			const size_t internalNumberOfZCells = memoryArea.getNumberOfZCells() - 2 * numberOfGhostCells - 1;
			
			const size_t blockSize = 1024;
			const size_t size = numberOfXCells * numberOfYCells * numberOfZCells;

			applyBoundaryConditionsDevice<BoundaryConditions, top, xDir, yDir, zDir>
				<<<(size + blockSize - 1)/blockSize, blockSize>>>
				(memoryArea, numberOfXCells, numberOfYCells, numberOfZCells, 
				internalNumberOfXCells, 
				internalNumberOfYCells,
				internalNumberOfZCells, 
				numberOfGhostCells);

		}

		template<class BoundaryConditions>
		void callApplyBoundaryConditions(memory::View<real>& memoryArea, size_t numberOfGhostCells) {
			applyBoundaryConditions<BoundaryConditions, 0, 1, 0, 0>(memoryArea, numberOfGhostCells);
			applyBoundaryConditions<BoundaryConditions, 1, 1, 0, 0>(memoryArea, numberOfGhostCells);

			if (memoryArea.getNumberOfYCells() > 1) {
				applyBoundaryConditions<BoundaryConditions, 0, 0, 1, 0>(memoryArea, numberOfGhostCells);
				applyBoundaryConditions<BoundaryConditions, 1, 0, 1, 0>(memoryArea, numberOfGhostCells);
			}
			if (memoryArea.getNumberOfZCells() > 1) {
				applyBoundaryConditions<BoundaryConditions, 0, 0, 0, 1>(memoryArea, numberOfGhostCells);
				applyBoundaryConditions<BoundaryConditions, 1, 0, 0, 1>(memoryArea, numberOfGhostCells);
			}
		}
	}
	///
	/// Constructs a new instance
	/// \param numberOfGhostCells the number of ghost cells on each side to use.
	///
	template<class BoundaryConditions>
	BoundaryCUDA<BoundaryConditions>::BoundaryCUDA(size_t numberOfGhostCells)
		: numberOfGhostCells(numberOfGhostCells) {

	}

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
	template<class BoundaryConditions>
	void BoundaryCUDA < BoundaryConditions > :: applyBoundaryConditions(volume::Volume& volume,
		const grid::Grid& grid) {
		for (size_t var = 0; var < volume.getNumberOfVariables(); ++var) {
			memory::View<real> memoryArea = volume.getScalarMemoryArea(var)->getView();
			callApplyBoundaryConditions<BoundaryConditions>(memoryArea, numberOfGhostCells);
		}
	}

	template class BoundaryCUDA < Neumann > ;
	template class BoundaryCUDA < Periodic >;
}
}
