#include "alsfvm/boundary/BoundaryCUDA.hpp"
#include "cuda.h"


namespace alsfvm { namespace boundary { 

	namespace {
		template<class BoundaryConditions, bool top, bool xDir, bool yDir, bool zDir>
		__global__ void applyBoundaryConditionsDevice(memory::View<real>& memoryArea, 
			const size_t numberOfXCells, const size_t numberOfYCells, const size_t numberOfZCells, const size_t numberOfGhostCells) {
			const size_t x = threadIdx.x + blockDim.x * blockIdx.x + xDir;
			const size_t y = threadIdx.y + blockDim.y * blockIdx.y + yDir;
			const size_t z = threadIdx.z + blockDim.z * blockIdx.z + zDir;



			if (x > numberOfXCells || y > numberOfYCells || z > numberOfZCells) {
				return;
			}
			

			for (size_t ghostCell = 1; ghostCell <= numberOfGhostCells; ++ghostCell) {
				BoundaryConditions::applyBoundary(memoryArea, x, y, z, ghostCell, top, xDir, yDir, zDir);
			}
		}

		template<class BoundaryConditions, bool top, bool xDir, bool yDir, bool zDir>
		void applyBoundaryConditions(memory::View<real>& memoryArea, const size_t numberOfGhostCells) {
			const size_t numberOfXCells = memoryArea.getNumberOfXCells() - 2 * xDir;
			const size_t numberOfYCells = memoryArea.getNumberOfYCells() - 2 * xDir;
			const size_t numberOfZCells = memoryArea.getNumberOfZCells() - 2 * xDir;
			const bool hasYDir = numberOfYCells > 1;
			const bool hasZDir = numberOfZCells > 1;
			const size_t blockSize = 1024;
			dim3 blockDim(blockSize, hasYDir ? 1 : blockSize, hasZDir ? 1 : blockSize);
			dim3 gridDim((numberOfXCells + blockSize - 1) / blockDim.x,
				(numberOfYCells + blockSize - 1) / blockDim.y,
				(numberOfZCells + blockSize - 1) / blockDim.z);

			applyBoundaryConditions<BoundaryConditions, top, xDir, yDir, zDir>(memoryArea, numberOfXCells, numberOfYCells, numberOfZCells, numberOfGhostCells);

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
			auto memoryArea = volume.getScalarMemoryArea(var)->getView();
			callApplyBoundaryConditions<BoundaryConditions>(memoryArea, numberOfGhostCells);
		}
	}
}
}
