#pragma once
#include "alsfvm/memory/View.hpp"

namespace alsfvm {
namespace boundary {

class Periodic {
    public:
        __device__ __host__ static void applyBoundary(alsfvm::memory::View<real>&
            memoryArea,
            int x, int y, int z, int boundaryCell, int numberOfGhostCells,
            bool top, bool xDir, bool yDir, bool zDir) {
            // NOTE: Here we keep everything as signed int, there is no
            // way we will reach the signed/unsigned limit anyway,
            // and here we do some subtraction, therefore
            // signed would have been dangerous.
            const int sign = top ? -1 : 1;
            const int nx = memoryArea.nx - 2 * numberOfGhostCells;
            const int ny = memoryArea.ny - 2 * numberOfGhostCells;
            const int nz = memoryArea.nz - 2 * numberOfGhostCells;
            memoryArea.at(x - sign * boundaryCell * xDir,
                y - sign * boundaryCell * yDir,
                z - sign * boundaryCell * zDir)
                = memoryArea.at( x + sign * (-boundaryCell + nx) * xDir,
                        y + sign * (-boundaryCell + ny) * yDir,
                        z + sign * (-boundaryCell + nz) * zDir);
        }

};
}
}
