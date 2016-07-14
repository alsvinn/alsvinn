#include "alsfvm/reconstruction/ReconstructionCPU.hpp"
#include "alsfvm/equation/burgers/Burgers.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/reconstruction/WENO2.hpp"
#include "alsfvm/reconstruction/WENOF2.hpp"
#include "alsfvm/error/Exception.hpp"

namespace alsfvm { namespace reconstruction {

template<class ReconstructionType, class Equation>
ReconstructionCPU<ReconstructionType, Equation>::ReconstructionCPU(const simulator::SimulatorParameters &simulatorParameters)
    : parameters(static_cast<const typename Equation::Parameters&>(simulatorParameters.getEquationParameters()))
{

}

template<class ReconstructionType, class Equation>
void ReconstructionCPU<ReconstructionType, Equation>::performReconstruction(const volume::Volume &inputVariables,
                                              size_t direction,
                                              size_t indicatorVariable,
                                              volume::Volume &leftOut,
                                              volume::Volume &rightOut)
{



    if (direction > 2) {
        THROW("Direction can only be 0, 1 or 2, was given: " << direction);
    }
    const ivec3 directionVector(direction == 0, direction == 1, direction == 2);

    // Now we go on to do the actual reconstruction, choosing the stencil for
    // each point.
    const size_t nx = inputVariables.getTotalNumberOfXCells();
    const size_t ny = inputVariables.getTotalNumberOfYCells();
    const size_t nz = inputVariables.getTotalNumberOfZCells();

    const size_t ngc = this->getNumberOfGhostCells();

    // Sanity check, we need at least ONE point in the interior.
    assert(int(nx) > 2 * directionVector.x * 2);
    assert((directionVector.y == 0) || (int(ny) > ngc * directionVector.y * 2));
    assert((directionVector.z == 0) || (int(nz) > ngc * directionVector.z * 2));

    const size_t startX = directionVector.x * (ngc - 1);
    const size_t startY = directionVector.y * (ngc - 1);
    const size_t startZ = directionVector.z * (ngc - 1);

    const size_t endX = nx - directionVector.x * (ngc - 1);
    const size_t endY = ny - directionVector.y * (ngc - 1);
    const size_t endZ = nz - directionVector.z * (ngc - 1);


    typename Equation::ConstViews viewIn(inputVariables);
    typename Equation::Views viewLeft(leftOut);
    typename Equation::Views viewRight(rightOut);

    Equation eq(parameters);
    for (size_t z = startZ; z < endZ; z++) {
#pragma omp parallel for
        for (size_t y = startY; y < endY; y++) {
            for (int x = startX; x < int(endX); x++) {
                ReconstructionType::reconstruct(eq, viewIn, x, y, z, viewLeft, viewRight,
                                                directionVector.x, directionVector.y,
                                                directionVector.z);
            }
        }
    }
}

template<class ReconstructionType, class Equation>
size_t ReconstructionCPU<ReconstructionType, Equation>::getNumberOfGhostCells()
{
    return ReconstructionType::getNumberOfGhostCells();
}

template class ReconstructionCPU<WENO2<equation::euler::Euler>, equation::euler::Euler>;
template class ReconstructionCPU<WENOF2<equation::euler::Euler>, equation::euler::Euler>;


template class ReconstructionCPU<WENO2<equation::burgers::Burgers>, equation::burgers::Burgers>;
}
}
