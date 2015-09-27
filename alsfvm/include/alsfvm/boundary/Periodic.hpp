#pragma once
#include "alsfvm/memory/View.hpp"

namespace alsfvm {
	namespace boundary {

		class Periodic {
		public:
			__device__ __host__ static void applyBoundary(alsfvm::memory::View<real>& memoryArea,
				size_t x, size_t y, size_t z, size_t boundaryCell, size_t numberOfGhostCells,
				bool top, bool xDir, bool yDir, bool zDir)
			{
				const int sign = top ? -1 : 1;
				const size_t nx = memoryArea.nx - 2 * numberOfGhostCells;
				const size_t ny = memoryArea.ny - 2 * numberOfGhostCells;
				const size_t nz = memoryArea.nz - 2 * numberOfGhostCells;
				memoryArea.at(x - sign * boundaryCell * xDir,
					y - sign * boundaryCell * yDir,
					z - sign * boundaryCell * zDir)
					= memoryArea.at( x + sign * (-boundaryCell + nx)* xDir,
					y + sign * (-boundaryCell + ny)* yDir,
					z + sign * (-boundaryCell + nz)* zDir);
			}

		};
	}
}
