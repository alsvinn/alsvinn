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

#include "alsfvm/functional/IntervalFunctionalWriter.hpp"
#include "alsutils/log.hpp"

namespace alsfvm {
namespace functional {

IntervalFunctionalWriter::IntervalFunctionalWriter(volume::VolumeFactory
    volumeFactory,
    io::WriterPointer writer,
    FunctionalPointer functional)
    : volumeFactory(volumeFactory),
      writer(writer),
      functional(functional) {

}

void IntervalFunctionalWriter::write(const volume::Volume& conservedVariables,
    const volume::Volume& extraVariables,
    const grid::Grid& grid,
    const simulator::TimestepInformation& timestepInformation) {
    if (!conservedVolume) {
        makeVolumes(grid);
    }

    conservedVolume->makeZero();
    extraVolume->makeZero();

    (*functional)(*conservedVolume, *extraVolume, conservedVariables,
        extraVariables, 1, grid);

    if (functionalSize == grid.getDimensions()) {
        writer->write(*conservedVolume, *extraVolume, grid,
            timestepInformation);
    } else {
        const ivec3 numberOfNodes = grid.getGlobalSize() / grid.getDimensions();
        grid::Grid modifiedGrid(grid.getOrigin(),
            grid.getTop(),
            functionalSize,
            grid.getBoundaryConditions(),
            grid.getGlobalPosition(),
            numberOfNodes * functionalSize);

        ALSVINN_LOG(INFO, "modifiedGrid.getDimensions().x = " <<
            modifiedGrid.getDimensions().x);
        ALSVINN_LOG(INFO, "conservedVolume.x = " <<
            conservedVolume->getSize().x);
        ALSVINN_LOG(INFO, "extraVolume.x = " <<
            extraVolume->getSize().x);

        writer->write(*conservedVolume, *extraVolume, modifiedGrid,
            timestepInformation);
    }


}

void IntervalFunctionalWriter::makeVolumes(const grid::Grid& grid) {
    functionalSize = functional->getFunctionalSize(grid);

    conservedVolume = volumeFactory.createConservedVolume(functionalSize.x,
            functionalSize.y, functionalSize.z, 0);
    conservedVolume->makeZero();

    extraVolume = volumeFactory.createExtraVolume(functionalSize.x,
            functionalSize.y, functionalSize.z, 0);
    extraVolume->makeZero();
}

}
}
