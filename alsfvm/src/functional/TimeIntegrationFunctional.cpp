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
    writer->write(*conservedVolume, *extraVolume, grid, timestepInformation);
}

void TimeIntegrationFunctional::makeVolumes(const grid::Grid &grid)
{
    const auto size = functional->getFunctionalSize(grid);

    conservedVolume = volumeFactory.createConservedVolume(size.x, size.y, size.z, 0);
    extraVolume = volumeFactory.createExtraVolume(size.x, size.y, size.z, 0);

}

}
}
