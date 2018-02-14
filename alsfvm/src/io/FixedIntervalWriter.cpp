#include "alsfvm/io/FixedIntervalWriter.hpp"
#include <iostream>
#include <algorithm>

namespace alsfvm {
namespace io {

FixedIntervalWriter::FixedIntervalWriter(alsfvm::shared_ptr<Writer>& writer,
    real timeInterval, real endTime)
    : writer(writer), timeInterval(timeInterval), endTime(endTime), numberSaved(0) {

}

void FixedIntervalWriter::write(const volume::Volume& conservedVariables,
    const volume::Volume& extraVariables, const grid::Grid& grid,
    const simulator::TimestepInformation& timestepInformation) {
    const real currentTime = timestepInformation.getCurrentTime();

    if (currentTime >= numberSaved * timeInterval) {
        writer->write(conservedVariables, extraVariables, grid, timestepInformation);
        numberSaved++;
    }

}

real FixedIntervalWriter::adjustTimestep(real dt,
    const simulator::TimestepInformation& timestepInformation) const {
    if (numberSaved > 0) {
        const real nextSaveTime = numberSaved * timeInterval;
        return std::min(dt, nextSaveTime - timestepInformation.getCurrentTime());
    } else {
        return dt;
    }
}

}
}
