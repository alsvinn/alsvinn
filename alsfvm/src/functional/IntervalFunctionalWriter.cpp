#include "alsfvm/functional/IntervalFunctionalWriter.hpp"

namespace alsfvm { namespace functional {

IntervalFunctionalWriter::IntervalFunctionalWriter(volume::VolumeFactory volumeFactory,
                                                   io::WriterPointer writer,
                                                   FunctionalPointer functional)
    : volumeFactory(volumeFactory),
      writer(writer),
      functional(functional)
{

}

void IntervalFunctionalWriter::write(const volume::Volume &conservedVariables,
                                     const volume::Volume &extraVariables,
                                     const grid::Grid &grid,
                                     const simulator::TimestepInformation &timestepInformation)
{
    if (!conservedVolume) {
        makeVolumes(grid);
    }

    conservedVolume->makeZero();
    extraVolume->makeZero();

    (*functional)(*conservedVolume, *extraVolume, conservedVariables, extraVariables, 1, grid);

    grid::Grid modifiedGrid(grid.getOrigin(), grid.getTop(), functionalSize);
    writer->write(*conservedVolume, *extraVolume, modifiedGrid, timestepInformation);
}

void IntervalFunctionalWriter::makeVolumes(const grid::Grid &grid)
{
    functionalSize = functional->getFunctionalSize(grid);

    conservedVolume = volumeFactory.createConservedVolume(functionalSize.x, functionalSize.y, functionalSize.z, 0);
    conservedVolume->makeZero();

    extraVolume = volumeFactory.createExtraVolume(functionalSize.x, functionalSize.y, functionalSize.z, 0);
    extraVolume->makeZero();
}

}
}
