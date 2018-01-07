#include "alsfvm/io/TimeIntegratedWriter.hpp"

namespace alsfvm { namespace io {

TimeIntegratedWriter::TimeIntegratedWriter(alsfvm::shared_ptr<Writer> &writer,
                                           real time,
                                           real timeRadius)
    : writer(writer), time(time), timeRadius(timeRadius)
{

}

void TimeIntegratedWriter::write(const volume::Volume &conservedVariables,
                                 const volume::Volume &extraVariables,
                                 const grid::Grid &grid,
                                 const simulator::TimestepInformation &timestepInformation)
{

    const real currentTime = timestepInformation.getCurrentTime();
    if (std::abs(currentTime - time) < timeRadius) {

        if (!integratedConservedVariables) {
            integratedConservedVariables = conservedVariables.makeInstance();
            integratedConservedVariables->makeZero();

            integratedExtraVariables = extraVariables.makeInstance();
            integratedExtraVariables->makeZero();
        }
        const real dt = currentTime - lastTime;




          integratedConservedVariables->addLinearCombination(1,
                                                            dt, conservedVariables,
                                                            0, conservedVariables,
                                                            0, conservedVariables,
                                                            0, conservedVariables);

         integratedExtraVariables->addLinearCombination(1,
                                                        dt, extraVariables,
                                                        0, extraVariables,
                                                        0, extraVariables,
                                                        0, extraVariables);


    }

    if (!written && currentTime >= time + timeRadius) {

        simulator::TimestepInformation newTime(time, 0);
        *integratedConservedVariables *= 0.5/timeRadius;
        *integratedExtraVariables *= 0.5/timeRadius;
        writer->write(*integratedConservedVariables, *integratedExtraVariables, grid, newTime);
        written = true;
    }
    lastTime = currentTime;
}

}
}
