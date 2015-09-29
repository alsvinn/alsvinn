#include "alsfvm/volume/EulerPrimitiveVolume.hpp"

namespace alsfvm { namespace volume {

EulerPrimitiveVolume::EulerPrimitiveVolume(boost::shared_ptr<memory::MemoryFactory> memoryFactory, size_t nx, size_t ny, size_t nz, size_t numberOfGhostCells)
    : Volume({ "rho", "ux", "uy", "uz", "p" }, memoryFactory, nx, ny, nz, numberOfGhostCells)
{
    // empty
}

}
}
