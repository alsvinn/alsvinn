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

#include "alsfvm/functional/TimeIntegrationFunctional.hpp"

namespace alsfvm {
namespace functional {

TimeIntegrationFunctional::TimeIntegrationFunctional(volume::VolumeFactory
    volumeFactory,
    io::WriterPointer writer,
    FunctionalPointer functional,
    double time,
    double timeRadius)
    : volumeFactory(volumeFactory), writer(writer), functional(functional),
      time(time), timeRadius(timeRadius) {

}

void TimeIntegrationFunctional::write(const volume::Volume& conservedVariables,
    const grid::Grid& grid,
    const simulator::TimestepInformation& timestepInformation) {
    if (!conservedVolume) {
        makeVolumes(grid);
    }

    const double currentTime = timestepInformation.getCurrentTime();
    const double dt = currentTime - lastTime;

    if (std::abs(currentTime - time) <= timeRadius) {

        (*functional)(*conservedVolume,  conservedVariables,
            dt, grid);
    }

    lastTime = currentTime;
}

void TimeIntegrationFunctional::finalize(const grid::Grid& grid,
    const simulator::TimestepInformation& timestepInformation) {
    grid::Grid smallerGrid(grid.getOrigin(),
        grid.getTop(),
        functionalSize,
        grid.getBoundaryConditions(),
        grid.getGlobalPosition(),
        grid.getGlobalSize());
    writer->write(*conservedVolume, smallerGrid, timestepInformation);
}

void TimeIntegrationFunctional::makeVolumes(const grid::Grid& grid) {
    functionalSize = functional->getFunctionalSize(grid);

    conservedVolume = volumeFactory.createConservedVolume(functionalSize.x,
            functionalSize.y, functionalSize.z, 0);
    conservedVolume->makeZero();


}

}
}
