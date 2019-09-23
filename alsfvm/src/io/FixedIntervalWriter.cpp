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

#include "alsfvm/io/FixedIntervalWriter.hpp"
#include <iostream>
#include <algorithm>

namespace alsfvm {
namespace io {

FixedIntervalWriter::FixedIntervalWriter(alsfvm::shared_ptr<Writer>& writer,
    real timeInterval, real, bool writeInitialTimestep, real startTime)
    : writer(writer), timeInterval(timeInterval), numberSaved(0),
      writeInitialTimestep(writeInitialTimestep), startTime(startTime) {
    if (!writeInitialTimestep) {
        numberSaved = 1;
    }

}

void FixedIntervalWriter::write(const volume::Volume& conservedVariables,
    const grid::Grid& grid,
    const simulator::TimestepInformation& timestepInformation) {
    const real currentTime = timestepInformation.getCurrentTime();

    if (currentTime >= numberSaved * timeInterval + startTime) {
        writer->write(conservedVariables, grid, timestepInformation);
        numberSaved++;
    }

}

real FixedIntervalWriter::adjustTimestep(real dt,
    const simulator::TimestepInformation& timestepInformation) const {
    if (numberSaved > 0) {
        const real nextSaveTime = numberSaved * timeInterval + startTime;
        return std::min(dt, nextSaveTime - timestepInformation.getCurrentTime());
    } else {
        return dt;
    }
}

void FixedIntervalWriter::finalize(const grid::Grid& grid,
    const simulator::TimestepInformation& timestepInformation) {
    writer->finalize(grid, timestepInformation);
}

}
}
