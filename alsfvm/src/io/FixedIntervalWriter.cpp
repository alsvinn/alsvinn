#include "alsfvm/io/FixedIntervalWriter.hpp"
#include <iostream>
#include <algorithm>

namespace alsfvm { namespace io {

FixedIntervalWriter::FixedIntervalWriter(alsfvm::shared_ptr<Writer> &writer,
                                         real timeInterval, real endTime)
    : writer(writer), timeInterval(timeInterval), endTime(endTime), numberSaved(0)
{

}

void FixedIntervalWriter::write(const volume::Volume &conservedVariables, const volume::Volume &extraVariables, const grid::Grid &grid, const simulator::TimestepInformation &timestepInformation)
{

    if (first) {
        dx = grid.getCellLengths().x;
        int nx = int(1/dx);

        if (nx%2 != 0) {
            THROW("Something wrong. N = " << nx);
        }

        N = nx/64;

        std::cout << "N = " << N << std::endl;

        numberSmallSaved = -N;

    }
    first = false;
    const real currentTime = timestepInformation.getCurrentTime();
    if (currentTime >= numberSaved * timeInterval + dx*(numberSmallSaved)) {
        writer->write(conservedVariables, extraVariables, grid, timestepInformation);
        std::cout << "Writing at " << timestepInformation.getCurrentTime() << std::endl;
        numberSmallSaved++;
    }

    if (numberSaved == 0) {
        numberSaved++;
    }
    if (numberSmallSaved == N+1)
    {
       numberSaved++;
       numberSmallSaved = -N;
    }

}

real FixedIntervalWriter::adjustTimestep(real dt, const simulator::TimestepInformation &timestepInformation) const
{
    if (numberSaved > 0) {
        const real nextSaveTime = numberSaved * timeInterval + dx*(numberSmallSaved);

        return std::min(dt, nextSaveTime - timestepInformation.getCurrentTime());
    } else {
        return dt;
    }
}

}
}
