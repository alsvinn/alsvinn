#include "alsfvm/io/FixedIntervalWriter.hpp"
#include <iostream>
namespace alsfvm { namespace io {

FixedIntervalWriter::FixedIntervalWriter(std::shared_ptr<Writer> &writer,
                                         real timeInterval, real endTime)
    : writer(writer), timeInterval(timeInterval), endTime(endTime), numberSaved(0)
{

}

void FixedIntervalWriter::write(const volume::Volume &conservedVariables, const volume::Volume &extraVariables, const grid::Grid &grid, const simulator::TimestepInformation &timestepInformation)
{
    const real currentTime = timestepInformation.getCurrentTime();
    if (numberSaved * timeInterval <= currentTime && currentTime < (numberSaved + 1) * timeInterval) {
        writer->write(conservedVariables, extraVariables, grid, timestepInformation);
        numberSaved++;
    }

}

}
}
