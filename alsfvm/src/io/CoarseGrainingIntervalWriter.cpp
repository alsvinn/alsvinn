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
    real endTime, int numberOfSkips)
    : writer(writer), timeInterval(timeInterval),
      numberOfCoarseSaves(numberOfCoarseSaves),
      numberOfSkips(numberOfSkips),
      endTime(endTime), numberSaved(0) {

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

}
}
