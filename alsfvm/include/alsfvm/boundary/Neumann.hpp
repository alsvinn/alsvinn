#pragma once
#include "alsfvm/memory/View.hpp"

namespace alsfvm {
namespace boundary {

class Neumann {
public:
	__device__ __host__ static void applyBoundary(alsfvm::memory::View<real>& memoryArea,
		size_t x, size_t y, size_t z, size_t boundaryCell, size_t numberOfGhostCells,
					   bool top, bool xDir, bool yDir, bool zDir)
    {
        const int sign = top ? -1 : 1;
        memoryArea.at(x - sign * boundaryCell * xDir,
                      y - sign * boundaryCell * yDir,
                      z - sign * boundaryCell * zDir )
                = memoryArea.at(x + sign * (boundaryCell - 1)* xDir,
                                y + sign * (boundaryCell - 1)* yDir,
                                z + sign * (boundaryCell - 1)* zDir);
    }

};
}
}
