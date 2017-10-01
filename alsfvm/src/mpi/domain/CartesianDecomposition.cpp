#include "alsfvm/mpi/domain/CartesianDecomposition.hpp"
#include "alsutils/error/Exception.hpp"
#include "alsfvm/mpi/CartesianCellExchanger.hpp"

namespace alsfvm { namespace mpi { namespace domain {

CartesianDecomposition::CartesianDecomposition(const DomainDecompositionParameters &parameters)
    : numberOfProcessors{parameters.getInteger("nx", 1),
                         parameters.getInteger("ny", 1),
                         parameters.getInteger("nz", 1)}
{

}

DomainInformationPtr CartesianDecomposition::decompose(ConfigurationPtr configuration,
                                                    const grid::Grid &grid)
{
    auto dimensions = grid.getDimensions();

    // Make sure we can evenly divide the dimensions. ie that
    // the number of processors to use in the x direction divides the number
    // of cells in the x direction, and so on.
    for (int i = 0; i < dimensions.size(); ++i) {
        if (dimensions[i]%numberOfProcessors[i] != 0) {
            THROW("Error in domain decompositon. In direction " << i<<"\n"
                  << "\tnumberOfProcessors assigned: " << numberOfProcessors[i]<<"\n"
                  <<"\tnumberOfCells assigned      : " << dimensions[i] <<"\n");
        }
    }

    int nodeNumber = configuration->getNodeNumber();
    // Find the x,y, z position of the nodeNumber.
    ivec3 nodePosition = ivec3{nodeNumber/(numberOfProcessors.y*numberOfProcessors.z),
        nodeNumber/(numberOfProcessors.z) % numberOfProcessors.y,
        nodeNumber % numberOfProcessors.z};

    ivec3 numberOfCellsPerProcessors = dimensions / numberOfProcessors;
    // startIndex for the grid
    ivec3 startIndex = (numberOfCellsPerProcessors) * nodePosition;

    // Geometrical position
    rvec3 startPosition = grid.getCellLengths() * startIndex;
    rvec3 endPosition = startPosition + grid.getCellLengths() * numberOfCellsPerProcessors;

    // Create new local grid.
    auto newGrid = alsfvm::make_shared<grid::Grid>(startPosition,
                                                   endPosition,
                                                   numberOfCellsPerProcessors);



    // Find neighbours:
    ivec6 neighbours;
    for (int side = 0; side < grid.getActiveDimension()*2; ++side) {
        if (nodePosition[side/2] == 0) {
            // we are on the boundary

            // We should only exchange if it is periodic
            if (grid.getBoundaryCondition(side) != boundary::Type::PERIODIC) {
                neighbours[side] = -1;
                continue;
            }
        }

        ivec3 neighbourPosition = nodePosition;
        neighbourPosition[side] -= 1;
        neighbourPosition[side] %= numberOfProcessors[side];


        int neighbourIndex = neighbourPosition.x +
                neighbourPosition.y * numberOfProcessors.x
                + neighbourPosition.z * numberOfProcessors.x * numberOfProcessors.y;

        neighbours[side] = neighbourIndex;
    }

    alsfvm::shared_ptr<CellExchanger> cellExchanger(new CartesianCellExchanger(configuration, neighbours));

    auto information = alsfvm::make_shared<DomainInformation>(newGrid, cellExchanger);

    return information;

}

}
}
}
