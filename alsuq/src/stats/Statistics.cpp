#include "alsuq/stats/Statistics.hpp"
#include "alsuq/mpi/utils.hpp"
namespace alsuq {
namespace stats {

void Statistics::write(const alsfvm::volume::Volume& conservedVariables,
    const alsfvm::volume::Volume& extraVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {
    computeStatistics(conservedVariables, extraVariables, grid,
        timestepInformation);
}



}
}
