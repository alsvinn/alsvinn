#pragma once
#include <array>
#include "alsfvm/types.hpp"
#include "alsfvm/reconstruction/Reconstruction.hpp"
#include "alsfvm/memory/MemoryFactory.hpp"

namespace alsfvm {
namespace reconstruction {

///
/// Performs ENO reconstruction of order order on the CPU.
///
/// See http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19980007543.pdf
///
template<int order>
class ENOCPU : public Reconstruction {
    public:
        ENOCPU(alsfvm::shared_ptr<memory::MemoryFactory>& memoryFactory,
            size_t nx, size_t ny, size_t nz);
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
        //! Creates the divided differences arrays
        //! @param nx number of x cells
        //! @param ny number of y cells
        //! @param nz number of z cells
        void makeDividedDifferenceArrays(size_t nx, size_t ny, size_t nz);

        void computeDividedDifferences(const memory::Memory<real>& input,
            const ivec3& direction,
            size_t level,
            memory::Memory<real>& output,
            const ivec3& start,
            const ivec3& end);

        alsfvm::shared_ptr<alsfvm::memory::MemoryFactory> memoryFactory;
        // For each level l, this will contain the divided differences for that
        // level.
        std::array < alsfvm::shared_ptr<memory::Memory<real> >,
            order - 1 > dividedDifferences;
};


} // namespace alsfvm
} // namespace reconstruction
