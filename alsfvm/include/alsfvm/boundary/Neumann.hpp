#pragma once
#include "alsfvm/memory/View.hpp"

namespace alsfvm {
namespace boundary {

class Neumann {
public:
    template<bool top, bool xDir, bool yDir, bool zDir>
    static void applyBoundary(alsfvm::memory::View& memoryArea,
                       size_t x, size_t y, size_t z, size_t boundaryCell)
    {
        const int sign = top ? 1 : -1;
        memoryArea.at(x - sign * boundaryCell * xDir,
                      y - sign * boundaryCell * yDir,
                      z - sign * boundaryCell * zDir )
                = memoryArea.at(x + sign * boundaryCell * xDir,
                                y + sign * boundaryCell * yDir,
                                z + sign * boundaryCell * zDir);
    }

};
}
}
