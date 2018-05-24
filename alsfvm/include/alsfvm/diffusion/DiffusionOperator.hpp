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
#include "alsfvm/volume/Volume.hpp"

namespace alsfvm {
namespace diffusion {

/// Applies numerical diffusion to the given conserved variables
///
/// This is typically used for the TeCNO-scheme, see
/// http://www.cscamm.umd.edu/tadmor/pub/TV+entropy/Fjordholm_Mishra_Tadmor_SINUM2012.pdf
class DiffusionOperator {
public:
    virtual ~DiffusionOperator() {};

    /// Applies numerical diffusion to the outputVolume given the data in conservedVolume.
    ///
    /// \note The numerical diffusion will be added to outputVolume, ie. the code will
    /// essentially work like
    /// \code{.cpp}
    /// outputVolume += diffusion(conservedVolume);
    /// \endcode
    virtual void applyDiffusion(volume::Volume& outputVolume,
        const volume::Volume& conservedVolume) = 0;


    /// Gets the total number of ghost cells this diffusion needs,
    /// this is typically governed by reconstruction algorithm.
    virtual size_t getNumberOfGhostCells() const = 0;

};
} // namespace alsfvm
} // namespace diffusion
