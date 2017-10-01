#include "alsfvm/mpi/CartesianCellExchanger.hpp"
#include "alsutils/mpi/mpi_types.hpp"

namespace alsfvm { namespace mpi {

CartesianCellExchanger::CartesianCellExchanger(ConfigurationPtr &configuration,
                                               const ivec6 &neighbours)
    : configuration(configuration), neighbours(neighbours)
{

}

bool CartesianCellExchanger::hasSide(int side) const
{
    return neighbours[side] > -1;
}

RequestContainer CartesianCellExchanger::exchangeCells(volume::Volume &outputVolume,
                                           const volume::Volume &inputVolume)
{
    if (datatypes.size() == 0) {
        createDataTypes(outputVolume);
    }

    const int dimensions = outputVolume.getDimensions();

    RequestContainer container;
    for (int side = 0; side < 2 * dimensions; ++side) {
        if (hasSide(side)) {
            for (int var = 0; var < inputVolume.getNumberOfVariables(); ++ var) {
                container.addRequest(Request::isend(*inputVolume[var],
                                                    1,
                                                    datatypes[side].indexedDatatype(),
                                                    neighbours[side],
                                                    0,
                                                    *configuration
                                                    ));

                container.addRequest(Request::ireceive(*outputVolume[var],
                                                       1,
                                                       datatypes[side].indexedDatatype(),
                                                       neighbours[side],
                                                       0,
                                                       *configuration
                                                       ));

            }
        }
    }

    return container;

}

void CartesianCellExchanger::createDataType(int side, const volume::Volume &volume)
{
    const int ghostCells = volume.getNumberOfGhostCells()[side / 2];

    const auto numberOfCellsPerDirection = volume.getSize();
    const int numberOfCellsInDirection = numberOfCellsPerDirection[side / 2];


    const int dimensions = volume.getDimensions();

    int numberOfSegments = 1;


    if ( side < 2) { // x side
        if (dimensions == 1) {
            numberOfSegments = 1;
        } else if (dimensions == 2) {
            numberOfSegments = numberOfCellsPerDirection.y;
        } else {
            numberOfSegments = numberOfCellsPerDirection.y*numberOfCellsPerDirection.z;
        }
    } else if (side < 4) { // y side
        if (dimensions == 2) {
            numberOfSegments = 1;
        } else {
            numberOfSegments = numberOfCellsPerDirection.z;
        }

    } else { // z side
        numberOfSegments = 1;
    }

    std::vector<int> displacements(numberOfSegments, 0);
    std::vector<int> lengths(numberOfSegments, 0);

    for (int i = 0; i < numberOfSegments; ++i) {
        if (dimensions == 1) {
            displacements[i] = side * (numberOfCellsPerDirection.x - ghostCells);
            lengths[i] = ghostCells;

        } else if(dimensions == 2) {
            if ( side < 2) {

                if (side > 0 || i > 0) {
                    displacements[i] = (numberOfCellsPerDirection.x);
                }
                if (i > 0) {
                    displacements[i] += displacements[i-1];
                }
                lengths[i] = ghostCells;
            } else {
                if (side > 2) { // bottom side does not have any displacement
                    // we only have two segments in dimension 2 for the y-direction,
                    // and the first one is 0 displacement, therefore, we do not add
                    // displacements[i-1]
                    displacements[i] = numberOfCellsPerDirection.x*numberOfCellsPerDirection.y
                            - numberOfCellsPerDirection.x*ghostCells;
                }
                lengths[i] = ghostCells*numberOfCellsPerDirection.x;
            }
        } else {
            if ( side < 2) {
                if (side > 0 || i > 0) {
                    displacements[i] = (numberOfCellsPerDirection.x);
                }
                if (i > 0) {
                    displacements[i] += displacements[i-1];
                }
                lengths[i] = ghostCells;
            } else if (side < 4) {
                if (side > 2 || i > 0) {
                    displacements[i] = numberOfCellsPerDirection.x*numberOfCellsPerDirection.y;
                }
                if (i > 0) {
                    displacements[i]+= displacements[i-1];
                }
                lengths[i] = ghostCells*numberOfCellsPerDirection.x;
            } else {

                // There is only one segment in the z direction, and it only needs
                // displacement if it is the back side
                if (side == 5) {
                    displacements[i] = numberOfCellsPerDirection.x*numberOfCellsPerDirection.y*numberOfCellsPerDirection.z-
                            numberOfCellsPerDirection.x*numberOfCellsPerDirection.y*ghostCells;
                }
                lengths[i] = ghostCells*numberOfCellsPerDirection.x*numberOfCellsPerDirection.y;
            }
        }
    }
    datatypes.push_back(MpiIndexType(numberOfSegments, lengths, displacements,
                                  alsutils::mpi::MpiTypes<real>::MPI_Real));
}

void CartesianCellExchanger::createDataTypes(const volume::Volume &volume)
{
    const int dimensions = volume.getDimensions();

    for (int side = 0; side < dimensions * 2; ++side) {
        createDataType(side, volume);
    }


}

}
}
