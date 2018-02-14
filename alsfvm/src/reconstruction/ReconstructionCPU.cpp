#include "alsfvm/reconstruction/ReconstructionCPU.hpp"
#include "alsfvm/equation/burgers/Burgers.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/reconstruction/WENO2.hpp"
#include "alsfvm/reconstruction/WENOF2.hpp"
#include "alsfvm/reconstruction/MC.hpp"
#include "alsutils/error/Exception.hpp"

namespace alsfvm {
namespace reconstruction {

template<class ReconstructionType, class Equation>
ReconstructionCPU<ReconstructionType, Equation>::ReconstructionCPU(
    const simulator::SimulatorParameters& simulatorParameters)
    : parameters(static_cast<const typename Equation::Parameters&>
          (simulatorParameters.getEquationParameters())) {

}

template<class ReconstructionType, class Equation>
void ReconstructionCPU<ReconstructionType, Equation>::performReconstruction(
    const volume::Volume& inputVariables,
    size_t direction,
    size_t indicatorVariable,
    volume::Volume& leftOut,
    volume::Volume& rightOut, const ivec3& start,
    const ivec3& end) {



    if (direction > 2) {
        THROW("Direction can only be 0, 1 or 2, was given: " << direction);
    }

    const ivec3 directionVector(direction == 0, direction == 1, direction == 2);

    // Now we go on to do the actual reconstruction, choosing the stencil for
    // each point.
    const int nx = inputVariables.getTotalNumberOfXCells();
    const int ny = inputVariables.getTotalNumberOfYCells();
    const int nz = inputVariables.getTotalNumberOfZCells();

    const int dimension = inputVariables.getDimensions();
    const int ngx = this->getNumberOfGhostCells();
    const int ngy = (dimension > 1) * this->getNumberOfGhostCells();
    const int ngz = (dimension > 2) * this->getNumberOfGhostCells();


    // Sanity check, we need at least ONE point in the interior.
    assert(int(nx) > 2 * directionVector.x * 2);
    assert((directionVector.y == 0) || (ny > ngy * directionVector.y * 2));
    assert((directionVector.z == 0) || (nz > ngz * directionVector.z * 2));

    const int startX = (ngx - directionVector.x * 1) + start.x;
    const int startY = (ngy - directionVector.y * 1) + start.y;
    const int startZ = (ngz - directionVector.z * 1) + start.z;

    const int endX = nx - (ngx - directionVector.x * 1) + end.x;
    const int endY = ny - (ngy - directionVector.y * 1) + end.y;
    const int endZ = nz - (ngz - directionVector.z * 1) + end.z;


    typename Equation::ConstViews viewIn(inputVariables);
    typename Equation::Views viewLeft(leftOut);
    typename Equation::Views viewRight(rightOut);

    Equation eq(parameters);

    for (int z = startZ; z < endZ; z++) {
        #pragma omp parallel for

        for (int y = startY; y < endY; y++) {
            for (int x = startX; x < int(endX); x++) {
                ReconstructionType::reconstruct(eq, viewIn, x, y, z, viewLeft, viewRight,
                    directionVector.x, directionVector.y,
                    directionVector.z);
            }
        }
    }
}

template<class ReconstructionType, class Equation>
size_t ReconstructionCPU<ReconstructionType, Equation>::getNumberOfGhostCells() {
    return ReconstructionType::getNumberOfGhostCells();
}

template class
ReconstructionCPU<WENO2 <equation::euler::Euler<1>>, equation::euler::Euler<1>>;
template class
ReconstructionCPU<WENOF2<equation::euler::Euler<1>>, equation::euler::Euler<1>>;
template class
ReconstructionCPU<MC<equation::euler::Euler<1>>, equation::euler::Euler<1>>;

template class
ReconstructionCPU<WENO2 <equation::euler::Euler<2>>, equation::euler::Euler<2>>;
template class
ReconstructionCPU<WENOF2<equation::euler::Euler<2>>, equation::euler::Euler<2>>;
template class
ReconstructionCPU<MC<equation::euler::Euler<2>>, equation::euler::Euler<2>>;

template class
ReconstructionCPU<WENO2 <equation::euler::Euler<3>>, equation::euler::Euler<3>>;
template class
ReconstructionCPU<WENOF2<equation::euler::Euler<3>>, equation::euler::Euler<3>>;
template class
ReconstructionCPU<MC<equation::euler::Euler<3>>, equation::euler::Euler<3>>;


template class
ReconstructionCPU<WENO2<equation::burgers::Burgers>, equation::burgers::Burgers>;
template class
ReconstructionCPU<MC<equation::burgers::Burgers>, equation::burgers::Burgers>;


}
}
