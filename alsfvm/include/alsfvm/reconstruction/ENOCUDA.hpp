/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once
#include <boost/array.hpp>
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
template<class Equation, int order>
class ENOCUDA : public Reconstruction {
public:
    ENOCUDA(alsfvm::shared_ptr<memory::MemoryFactory>& memoryFactory,
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
    void computeDividedDifferences(const memory::Memory<real>& input,
        const ivec3& direction,
        size_t level,
        memory::Memory<real>& output,
        const ivec3& startIndex,
        const ivec3& endIndex);

    // For each level l, this will contain the divided differences for that
    // level.
    std::array < alsfvm::shared_ptr<memory::Memory<real> >,
        order - 1 > dividedDifferences;
};


} // namespace alsfvm
} // namespace reconstruction
