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
#include "alsfvm/mpi/RequestContainer.hpp"
#include "alsfvm/integrator/WaveSpeedAdjuster.hpp"


namespace alsfvm {
namespace mpi {

//! Abstract base class for exchanging cells
class CellExchanger : public integrator::WaveSpeedAdjuster {
public:

    virtual ~CellExchanger() {}
    //! Does the exchange of cells between the processors.
    virtual RequestContainer exchangeCells(alsfvm::volume::Volume& outputVolume,
        const alsfvm::volume::Volume& inputVolume) = 0;

    //! Does the maximum over all processors
    virtual real max(real number) = 0;

    //! Does the maximum over all wave speeds across processors
    real adjustWaveSpeed(real waveSpeed);

    virtual ivec6 getNeighbours() const = 0;
};

typedef alsfvm::shared_ptr<CellExchanger> CellExchangerPtr;
} // namespace mpi
} // namespace alsfvm
