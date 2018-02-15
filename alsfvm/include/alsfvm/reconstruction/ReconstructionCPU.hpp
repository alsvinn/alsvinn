#pragma once
#include "alsfvm/reconstruction/Reconstruction.hpp"
#include "alsfvm/simulator/SimulatorParameters.hpp"

namespace alsfvm {
namespace reconstruction {

///
/// This is meant to do arbitrary reconstruction on the CPU, given a suitable
/// template argument
///
template<class ReconstructionType, class Equation>
class ReconstructionCPU : public Reconstruction {
public:
    ReconstructionCPU(const simulator::SimulatorParameters& simulatorParameters);
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
    /// \param[in] start (positive) the first index to compute the flux for
    /// \param[in] end (negative) the offset to on the upper part of the grid
    ///
    virtual void performReconstruction(const volume::Volume& inputVariables,
        size_t direction,
        size_t indicatorVariable,
        volume::Volume& leftOut,
        volume::Volume& rightOut, const ivec3& start = {0, 0, 0},
        const ivec3& end = {0, 0, 0});

    ///
    /// \brief getNumberOfGhostCells returns the number of ghost cells we need
    ///        for this computation
    /// \return order.
    ///
    virtual size_t getNumberOfGhostCells();
private:
    typename Equation::Parameters parameters;
};
} // namespace alsfvm
} // namespace reconstruction
