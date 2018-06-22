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

#include "alsfvm/io/CoarseGrainingIntervalWriter.hpp"
#include <iostream>
#include <algorithm>
#include "alsutils/log.hpp"

namespace alsfvm {
namespace io {

CoarseGrainingIntervalWriter::CoarseGrainingIntervalWriter(
    alsfvm::shared_ptr<Writer>& writer,
    real timeInterval,
    int numberOfCoarseSaves,
    real, int numberOfSkips)
    : writer(writer), timeInterval(timeInterval),
      numberOfCoarseSaves(numberOfCoarseSaves),
      numberOfSkips(numberOfSkips),
      numberSaved(0) {

}

void CoarseGrainingIntervalWriter::write(const volume::Volume&
    conservedVariables, const volume::Volume& extraVariables,
    const grid::Grid& grid, const simulator::TimestepInformation&
    timestepInformation) {


    dx = grid.getCellLengths().x;
    const real currentTime = timestepInformation.getCurrentTime();

    if (currentTime >= numberSaved * timeInterval + (numberOfSkips + 1)*dx *
        (numberSmallSaved)) {
        writer->write(conservedVariables, extraVariables, grid, timestepInformation);
        ALSVINN_LOG(INFO, "Writing at " << timestepInformation.getCurrentTime()
            << "("
            << "\tnumberSaved = " << numberSaved << "\n"
            << "\ttimeInterval = " << timeInterval << "\n"
            << "\tdx = " << dx << "\n"
            << "\tnumberSmallSaved = " << numberSmallSaved << "\n"
            << "\tnumberOfCoarseSaves = " << numberOfCoarseSaves << "\n"
            << ")");
        numberSmallSaved++;
    }

    if (numberSaved == 0) {
        numberSaved++;
    }


    if (first) {
        numberSmallSaved = -numberOfCoarseSaves;
    }

    first = false;

    if (numberSmallSaved == numberOfCoarseSaves + 1) {
        numberSaved++;
        numberSmallSaved = -numberOfCoarseSaves;
    }

}

real CoarseGrainingIntervalWriter::adjustTimestep(real dt,
    const simulator::TimestepInformation& timestepInformation) const {

    if (numberSaved > 0) {
        const real nextSaveTime = numberSaved * timeInterval + (numberOfSkips + 1) *
            dx * (numberSmallSaved);

        return std::min(dt, nextSaveTime - timestepInformation.getCurrentTime());
    } else {
        return dt;
    }
}

void CoarseGrainingIntervalWriter::finalize(const grid::Grid& grid,
    const simulator::TimestepInformation& timestepInformation) {
    writer->finalize(grid, timestepInformation);
}

}
}
