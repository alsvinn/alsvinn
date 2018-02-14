#pragma once
#include "alsfvm/reconstruction/Reconstruction.hpp"
namespace alsfvm {
namespace reconstruction {

class NoReconstructionCUDA : public Reconstruction {
    public:

        ///
        /// Copies the data to the other arrays. Does no reconstruction.
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
        virtual void performReconstruction(const volume::Volume& inputVariables,
            size_t direction,
            size_t indicatorVariable,
            volume::Volume& leftOut,
            volume::Volume& rightOut, const ivec3& start = {0, 0, 0},
            const ivec3& end = {0, 0, 0});

        size_t getNumberOfGhostCells();


};
} // namespace alsfvm
} // namespace reconstruction
