#include "alsfvm/reconstruction/NoReconstruction.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
namespace alsfvm {
namespace reconstruction {
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
void NoReconstruction::performReconstruction(const volume::Volume&
    inputVariables,
    size_t direction,
    size_t indicatorVariable,
    volume::Volume& leftOut,
    volume::Volume& rightOut, const ivec3& start,
    const ivec3& end) {
    for (size_t var = 0; var < inputVariables.getNumberOfVariables(); var++) {
        auto pointerIn = inputVariables.getScalarMemoryArea(var)->getPointer();
        auto pointerLeft = leftOut.getScalarMemoryArea(var)->getPointer();
        auto pointerRight = rightOut.getScalarMemoryArea(var)->getPointer();

        volume::for_each_cell_index(inputVariables,
        [&](size_t index) {

            pointerLeft[index] = pointerIn[index];
            pointerRight[index] = pointerIn[index];
        });
    }

}

///
/// \brief getNumberOfGhostCells returns the number of ghost cells we need
///        for this computation
/// \return order.
///
size_t NoReconstruction::getNumberOfGhostCells() {
    return 1;
}
}
}
