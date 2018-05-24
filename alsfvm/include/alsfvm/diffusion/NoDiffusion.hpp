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
#include "alsfvm/diffusion/DiffusionOperator.hpp"
namespace alsfvm {
namespace diffusion {

//! Null object for diffusion. Doesn't actually add any diffusion
//! at all.
class NoDiffusion : public DiffusionOperator {
public:
    virtual void applyDiffusion(volume::Volume& outputVolume,
        const volume::Volume& conservedVolume) ;


    /// Gets the total number of ghost cells this diffusion needs,
    /// this is typically governed by reconstruction algorithm.
    virtual size_t getNumberOfGhostCells() const;
};
} // namespace diffusion
} // namespace alsfvm
