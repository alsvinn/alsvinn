#pragma once
#include "alsfvm/reconstruction/Reconstruction.hpp"
namespace alsfvm { namespace reconstruction { 

///
/// \brief The NoReconstruction class is the default reconstruction option (eg. none)
///
/// Here we do not perform any reconstruction, we simply copy the data into the
/// correct arrays, with correct indexing.
///
    class NoReconstruction : public Reconstruction{
    public:

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
        /// \param[in] indicatorVariable is ignored by this implementation
        /// \param[out] leftOut at the end, will contain the left values
        ///                     for all grid cells in the interior.
        ///
        /// \param[out] rightOut at the end, will contain the right values
        ///                     for all grid cells in the interior.
        ///
        /// \todo This can be done more efficiently, but we will wait with this.
        ///
        virtual void performReconstruction(const volume::Volume& inputVariables,
                                   size_t direction,
                                   size_t indicatorVariable,
                                   volume::Volume& leftOut,
                                           volume::Volume& rightOut, const ivec3& start={0,0,0},
                                           const ivec3& end={0,0,0});

        ///
        /// \brief getNumberOfGhostCells returns the number of ghost cells we need
        ///        for this computation
        /// \returns 1.
        ///
        virtual size_t getNumberOfGhostCells();

    };
} // namespace alsfvm
} // namespace reconstruction
