#include "alsfvm/reconstruction/ReconstructionCUDA.hpp"
#include "alsfvm/reconstruction/WENOF2.hpp"
#include "alsfvm/reconstruction/WENO2.hpp"
#include "alsfvm/reconstruction/MC.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/equation/burgers/Burgers.hpp"
#include "alsfvm/cuda/cuda_utils.hpp"


namespace alsfvm {
namespace reconstruction {
namespace {


template<class Equation, class ReconstructionType,  size_t dimension, bool xDir, bool yDir, bool zDir>
__global__ void reconstructDevice(Equation eq, typename Equation::ConstViews input,
                                  typename Equation::Views left, typename Equation::Views right,

                                  size_t numberOfXCells, size_t numberOfYCells, size_t numberOfZCells,
                                  ivec3 start, ivec3 end) {

    const size_t index = threadIdx.x + blockDim.x * blockIdx.x;
    // We have
    // index = z * nx * ny + y * nx + x;
    const size_t xInternalFormat = index % numberOfXCells;
    const size_t yInternalFormat = (index / numberOfXCells) % numberOfYCells;
    const size_t zInternalFormat = (index) / (numberOfXCells * numberOfYCells);

    if (xInternalFormat >= numberOfXCells || yInternalFormat >= numberOfYCells || zInternalFormat >= numberOfZCells) {
        return;
    }
    const size_t x = xInternalFormat + (ReconstructionType::getNumberOfGhostCells()-1) + start.x;
    const size_t y = yInternalFormat + (dimension>1)*(ReconstructionType::getNumberOfGhostCells()-1) + start.y;
    const size_t z = zInternalFormat + (dimension>2)*(ReconstructionType::getNumberOfGhostCells()-1) + start.z;

    ReconstructionType::reconstruct(eq, input, x, y, z, left, right, xDir, yDir, zDir);


}

template<class Equation, class ReconstructionType, size_t dimension, bool xDir, bool yDir, bool zDir >
void callReconstructionDevice(const Equation& equation, const volume::Volume& inputVariables,
                              size_t direction,
                              size_t indicatorVariable,
                              volume::Volume& leftOut,
                              volume::Volume& rightOut,
                              const ivec3& start,
                              const ivec3& end) {

    const int numberOfXCells = int(leftOut.getTotalNumberOfXCells()) - 2 * (ReconstructionType::getNumberOfGhostCells()-1) - start.x + end.x;
    const int numberOfYCells = int(leftOut.getTotalNumberOfYCells()) - 2 * (dimension>1)*(ReconstructionType::getNumberOfGhostCells()-1) - start.y + end.y;
    const int numberOfZCells = int(leftOut.getTotalNumberOfZCells()) - 2 * (dimension>2)*(ReconstructionType::getNumberOfGhostCells()-1) - start.z + end.z;

    const size_t totalSize = numberOfXCells * numberOfYCells * numberOfZCells;


    const size_t blockSize = 512;
    const size_t gridSize = (totalSize + blockSize - 1) / blockSize;



    typename Equation::Views viewLeft(leftOut);
    typename Equation::Views viewRight(rightOut);
    typename Equation::ConstViews viewInput(inputVariables);
        reconstructDevice<Equation, ReconstructionType, dimension, xDir, yDir, zDir> << <gridSize, blockSize >> >(equation, viewInput,
                                                                                                              viewLeft, viewRight,
                                                                                                              numberOfXCells,
                                                                                                              numberOfYCells,
                                                                                                              numberOfZCells,
                                                                                                              start, end);



}

template<size_t dimension, class Equation, class ReconstructionType>
void performReconstructionDevice(const Equation& equation, const volume::Volume& inputVariables,
                                 size_t direction,
                                 size_t indicatorVariable,
                                 volume::Volume& leftOut,
                                 volume::Volume& rightOut,
                                 const ivec3& start,
                                 const ivec3& end) {
    assert(direction < 3);
    switch (direction) {
    case 0:
        callReconstructionDevice<Equation, ReconstructionType, dimension, 1, 0, 0>(equation,
                                                                                   inputVariables,
                                                                                   direction,
                                                                                   indicatorVariable,
                                                                                   leftOut,
                                                                                   rightOut,
                                                                                   start,
                                                                                   end);
        break;

    case 1:
        callReconstructionDevice<Equation, ReconstructionType, dimension, 0, 1, 0>(equation,
                                                                                   inputVariables,
                                                                                   direction,
                                                                                   indicatorVariable,
                                                                                   leftOut,
                                                                                   rightOut,
                                                                                   start,
                                                                                   end);
        break;

    case 2:
        callReconstructionDevice<Equation, ReconstructionType, dimension, 0, 0, 1>(equation,
                                                                                   inputVariables,
                                                                                   direction,
                                                                                   indicatorVariable,
                                                                                   leftOut,
                                                                                   rightOut,
                                                                                   start,
                                                                                   end);
        break;
    }

}

}



///
/// Performs reconstruction.
/// \param[in] inputVariables the variables to reconstruct.
/// \param[in] direction the direction:
/// direction | explanation
/// ----------|------------
///     0     |   x-direction
///     1     |   y-direction
///     2     |   z-direction
///
/// \param[in] indicatorVariable the variable number to use for
/// stencil selection. We will determine the stencil based on
/// inputVariables->getScalarMemoryArea(indicatorVariable).
///
/// \param[out] leftOut at the end, will contain the left interpolated values
///                     for all grid cells in the interior.
///
/// \param[out] rightOut at the end, will contain the right interpolated values
///                     for all grid cells in the interior.
///
template<class ReconstructionType, class Equation>
void ReconstructionCUDA<ReconstructionType, Equation>::performReconstruction(const volume::Volume& inputVariables,
                                                                             size_t direction,
                                                                             size_t indicatorVariable,
                                                                             volume::Volume& leftOut,
                                                                             volume::Volume& rightOut, const ivec3& start,
                                                                             const ivec3& end)
{
    size_t dimension = 1 + (leftOut.getNumberOfYCells() > 1) + (leftOut.getNumberOfZCells() > 1);

    switch (dimension) {
    case 1:
        performReconstructionDevice<1, Equation, ReconstructionType>(equation,
                                                                     inputVariables,
                                                                     direction,
                                                                     indicatorVariable,
                                                                     leftOut,
                                                                     rightOut,
                                                                     start,
                                                                     end);
        break;
    case 2:
        performReconstructionDevice<2, Equation, ReconstructionType>(equation,
                                                                     inputVariables,
                                                                     direction,
                                                                     indicatorVariable,
                                                                     leftOut,
                                                                     rightOut,
                                                                     start,
                                                                     end);
        break;
    case 3:
        performReconstructionDevice<3, Equation, ReconstructionType>(equation,
                                                                     inputVariables,
                                                                     direction,
                                                                     indicatorVariable,
                                                                     leftOut,
                                                                     rightOut,
                                                                     start,
                                                                     end);
        break;
    }

}

template<class ReconstructionType, class Equation>
size_t ReconstructionCUDA<ReconstructionType, Equation>::getNumberOfGhostCells()  {
    return ReconstructionType::getNumberOfGhostCells();
}

template<class ReconstructionType, class Equation>
ReconstructionCUDA<ReconstructionType, Equation>::ReconstructionCUDA(const simulator::SimulatorParameters& parameters)
    : equation(static_cast<const typename Equation::Parameters&>(parameters.getEquationParameters()))
{


}

template class ReconstructionCUDA < WENOF2<equation::euler::Euler<1> >, equation::euler::Euler<1> >;
template class ReconstructionCUDA <  WENO2<equation::euler::Euler<1> >, equation::euler::Euler<1> >;
template class ReconstructionCUDA <  MC<equation::euler::Euler<1> >, equation::euler::Euler<1> >;


template class ReconstructionCUDA < WENOF2<equation::euler::Euler<2> >, equation::euler::Euler<2> >;
template class ReconstructionCUDA <  WENO2<equation::euler::Euler<2> >, equation::euler::Euler<2> >;
template class ReconstructionCUDA <  MC<equation::euler::Euler<2> >, equation::euler::Euler<2> >;

template class ReconstructionCUDA < WENOF2<equation::euler::Euler<3> >, equation::euler::Euler<3> >;
template class ReconstructionCUDA <  WENO2<equation::euler::Euler<3> >, equation::euler::Euler<3> >;
template class ReconstructionCUDA <  MC<equation::euler::Euler<3> >, equation::euler::Euler<3> >;

template class ReconstructionCUDA < WENO2<equation::burgers::Burgers>, equation::burgers::Burgers>;
template class ReconstructionCUDA < MC<equation::burgers::Burgers>, equation::burgers::Burgers>;
}
}
