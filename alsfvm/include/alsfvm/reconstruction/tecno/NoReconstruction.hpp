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
#include "alsfvm/reconstruction/tecno/TecnoReconstruction.hpp"
namespace alsfvm {
namespace reconstruction {
namespace tecno {

//! Does no reconstruction, just copies the variables to the new struct.
class NoReconstruction : public TecnoReconstruction {
public:

    //! Copies the variables to the new arrays.
    //!
    //! @param[in] leftInput the left values to use for reconstruction
    //! @param[in] rightInput the right values to use for reconstruction
    //! @param[in] direction the direction (0=x, 1=y, 2=y)
    //! @param[out] leftOutput at the end, should contain reconstructed values
    //! @param[out] rightOutput at the end, should contain the reconstructed values
    virtual void performReconstruction(const volume::Volume& leftInput,
        const volume::Volume& rightInput,
        size_t direction,
        volume::Volume& leftOutput,
        volume::Volume& rightOutput);


    virtual size_t getNumberOfGhostCells() const {
        return 1;
    }
};
} // namespace tecno
} // namespace reconstruction
} // namespace alsfvm
