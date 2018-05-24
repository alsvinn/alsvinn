/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "alsuq/stats/StructureBasicCUDA.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsuq/stats/stats_util.hpp"
namespace alsuq {
namespace stats {


namespace {

//! Computes the structure function for FIXED h
//!
//! The goal is to compute the structure function, then reduce (sum) over space
//! then go on to next h
__global__ void computeStructureBasic(real* output,
    alsfvm::memory::View<const real> input,
    ivec3 directionVector,
    int h,
    int nx, int ny, int nz, int ngx, int ngy, int ngz,
    real p) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= nx * ny * nz) {
        return;
    }

    const int x = index % nx;
    const int y = (index / nx) % ny;
    const int z = (index / nx / ny);



    const int xNext = x + h * directionVector.x;
    const int yNext = y + h * directionVector.y;
    const int zNext = z + h * directionVector.z;

    const real u = input.at(x + ngx, y + ngy, z + ngz);
    const real u_h = input.at(xNext % nx + ngx, yNext % ny + ngy, zNext % nz + ngz);

    output[index] = pow(fabs(u - u_h), p);
}
}

StructureBasicCUDA::StructureBasicCUDA(const StatisticsParameters& parameters)
    : StatisticsHelper(parameters),
      direction(parameters.getInteger("direction")),
      p(parameters.getDouble("p")),
      directionVector(make_direction_vector(direction)),
      numberOfH(parameters.getInteger("numberOfH")),
      statisticsName ("structure_basic_" + std::to_string(p))

{

}

std::vector<std::string> StructureBasicCUDA::getStatisticsNames() const {
    return {statisticsName};
}

void StructureBasicCUDA::computeStatistics(const alsfvm::volume::Volume&
    conservedVariables,
    const alsfvm::volume::Volume& extraVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {
    auto& structure = this->findOrCreateSnapshot(statisticsName,
            timestepInformation,
            conservedVariables, extraVariables,
            numberOfH, 1, 1, "cpu");


    computeStructure(*structure.getVolumes().getConservedVolume(),
        conservedVariables);
    computeStructure(*structure.getVolumes().getExtraVolume(),
        extraVariables);
}

void StructureBasicCUDA::finalizeStatistics() {

}

void StructureBasicCUDA::computeStructure(alsfvm::volume::Volume& output,
    const alsfvm::volume::Volume& input) {

    for (size_t var = 0; var < input.getNumberOfVariables(); ++var) {
        auto inputView = input[var]->getView();
        auto outputView = output[var]->getView();

        const int ngx = input.getNumberOfXGhostCells();
        const int ngy = input.getNumberOfYGhostCells();
        const int ngz = input.getNumberOfZGhostCells();

        const int nx = int(input.getNumberOfXCells()) - 2 * ngx;
        const int ny = int(input.getNumberOfYCells()) - 2 * ngy;
        const int nz = int(input.getNumberOfZCells()) - 2 * ngz;

        const int dimensions = input.getDimensions();

        structureOutput.resize(nx * ny * nz, 0);

        for (int h = 1; h < int(numberOfH); ++h) {
            const int threads = 1024;
            const int size = nx * ny * nz;
            const int blockNumber = (size + threads - 1) / threads;

            computeStructureBasic <<< blockNumber, threads>>>(thrust::raw_pointer_cast(
                    structureOutput.data()), inputView, directionVector,
                h, nx, ny, nz, ngx, ngy, ngz, p);

            real structureResult = thrust::reduce(structureOutput.begin(),
                    structureOutput.end(),
                    0.0, thrust::plus<real>());

            outputView.at(h) += structureResult / (nx * ny * nz);
        }



    }
}
REGISTER_STATISTICS(cuda, structure_basic, StructureBasicCUDA)
}
}
