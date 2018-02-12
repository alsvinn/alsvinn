#include "alsfvm/functional/TimeIntegrationFunctional.hpp"

namespace alsfvm { namespace functional {

TimeIntegrationFunctional::TimeIntegrationFunctional(volume::VolumeFactory volumeFactory,
                                                     io::WriterPointer writer,
                                                     FunctionalPointer functional,
                                                     double time,
                                                     double timeRadius)
    : volumeFactory(volumeFactory), writer(writer), functional(functional),
      time(time), timeRadius(timeRadius)
{

}

void TimeIntegrationFunctional::write(const volume::Volume &conservedVariables,
                                      const volume::Volume &extraVariables,
                                      const grid::Grid &grid,
                                      const simulator::TimestepInformation &timestepInformation)
{
    if (!conservedVolume) {
        makeVolumes(grid);
    }

    const double currentTime = timestepInformation.getCurrentTime();
    const double dt = currentTime - lastTime;
    if (std::abs(currentTime-time) <= timeRadius) {

        (*functional)(*conservedVolume, *extraVolume, conservedVariables, extraVariables, dt, grid);
    }

    lastTime = currentTime;
}

void TimeIntegrationFunctional::finalize(const grid::Grid &grid,
                                         const simulator::TimestepInformation &timestepInformation)
{
    grid::Grid smallerGrid(grid.getOrigin(),
                        grid.getTop(),
                        functionalSize,
                        grid.getBoundaryConditions(),
                       grid.getGlobalPosition(),
                       grid.getGlobalSize());
    writer->write(*conservedVolume, *extraVolume, smallerGrid, timestepInformation);
}

void TimeIntegrationFunctional::makeVolumes(const grid::Grid &grid)
{
    functionalSize = functional->getFunctionalSize(grid);

    conservedVolume = volumeFactory.createConservedVolume(functionalSize.x, functionalSize.y, functionalSize.z, 0);
    conservedVolume->makeZero();

    extraVolume = volumeFactory.createExtraVolume(functionalSize.x, functionalSize.y, functionalSize.z, 0);
    extraVolume->makeZero();

}

}
}
