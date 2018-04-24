#include "alsfvm/functional/IntervalFunctionalWriter.hpp"
#include "alsutils/log.hpp"

namespace alsfvm {
namespace functional {

IntervalFunctionalWriter::IntervalFunctionalWriter(volume::VolumeFactory
    volumeFactory,
    io::WriterPointer writer,
    FunctionalPointer functional)
    : volumeFactory(volumeFactory),
      writer(writer),
      functional(functional) {

}

void IntervalFunctionalWriter::write(const volume::Volume& conservedVariables,
    const volume::Volume& extraVariables,
    const grid::Grid& grid,
    const simulator::TimestepInformation& timestepInformation) {
    if (!conservedVolume) {
        makeVolumes(grid);
    }

    conservedVolume->makeZero();
    extraVolume->makeZero();

    (*functional)(*conservedVolume, *extraVolume, conservedVariables,
        extraVariables, 1, grid);

    if (functionalSize == grid.getDimensions()) {
        writer->write(*conservedVolume, *extraVolume, grid,
            timestepInformation);
    } else {
        const ivec3 numberOfNodes = grid.getGlobalSize() / grid.getDimensions();
        grid::Grid modifiedGrid(grid.getOrigin(),
            grid.getTop(),
            functionalSize,
            grid.getBoundaryConditions(),
            grid.getGlobalPosition(),
            numberOfNodes * functionalSize);

        ALSVINN_LOG(INFO, "modifiedGrid.getDimensions().x = " <<
            modifiedGrid.getDimensions().x);
        ALSVINN_LOG(INFO, "conservedVolume.x = " <<
            conservedVolume->getSize().x);
        ALSVINN_LOG(INFO, "extraVolume.x = " <<
            extraVolume->getSize().x);

        writer->write(*conservedVolume, *extraVolume, modifiedGrid,
            timestepInformation);
    }


}

void IntervalFunctionalWriter::makeVolumes(const grid::Grid& grid) {
    functionalSize = functional->getFunctionalSize(grid);

    conservedVolume = volumeFactory.createConservedVolume(functionalSize.x,
            functionalSize.y, functionalSize.z, 0);
    conservedVolume->makeZero();

    extraVolume = volumeFactory.createExtraVolume(functionalSize.x,
            functionalSize.y, functionalSize.z, 0);
    extraVolume->makeZero();
}

}
}
