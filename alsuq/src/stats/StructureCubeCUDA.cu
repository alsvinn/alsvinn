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

#include "alsuq/stats/StructureCubeCUDA.hpp"

#include "alsfvm/volume/volume_foreach.hpp"
#include "alsuq/stats/stats_util.hpp"
#include "alsuq/stats/structure_common.hpp"
namespace alsuq {
namespace stats {

namespace {

// This will be nicer when we can finally upgrade to C++14
template<int p>
struct FastPower {
    __device__ __host__  static double power(double x, double) {
        return power_internal(x);
    }

    __device__ __host__ static double power_internal(double x);
};

template<int p>
__device__ __host__  double FastPower<p>::power_internal(double x){
    return x*FastPower<p-1>::power_internal(x);
}

template<>
__device__ __host__  double FastPower<1>::power_internal(double x) {
    return x;
}


struct PowfPower {
    __device__ __host__ static double power(double x, double p) {
        return powf(x, p);
    }
};

//! Computes the structure function for FIXED h
//!
//! The goal is to compute the structure function, then reduce (sum) over space
//! then go on to next h
//!
template<class PowerClass>
__global__ void computeStructureCube(real* output,
    alsfvm::memory::View<const real> input,
    int h,
    int nx, int ny, int nz, int ngx, int ngy, int ngz,
    real p, int dimensions) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= nx * ny * nz) {
        return;
    }

    const int i = index % nx;
    const int j = (index / nx) % ny;
    const int k = (index / nx / ny);


    output[index] = 0;
    forEachPointInComputeStructureCube([&] (double u, double u_h) {
        output[index] += PowerClass::power(fabs(u - u_h), p);
    }, input, i, j, k, h, nx, ny, nz, ngx, ngy, ngz, dimensions);

}

}

StructureCubeCUDA::StructureCubeCUDA(const StatisticsParameters& parameters)
    : StatisticsHelper(parameters),
      p(parameters.getDouble("p")),
      numberOfH(parameters.getInteger("numberOfH")),
      statisticsName ("structure_cube_" + std::to_string(p))

{

}

std::vector<std::string> StructureCubeCUDA::getStatisticsNames() const {
    return {statisticsName};
}

void StructureCubeCUDA::computeStatistics(const alsfvm::volume::Volume&
    conservedVariables,
    const alsfvm::volume::Volume& extraVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {
    auto& structure = this->findOrCreateSnapshot(statisticsName,
            timestepInformation,
            conservedVariables, extraVariables,
            numberOfH, 1, 1, "cpu");


    if (p == 1) {
        computeStructure<FastPower<1>>(*structure.getVolumes().getConservedVolume(),
        conservedVariables);
        computeStructure<FastPower<1>>(*structure.getVolumes().getExtraVolume(),
        extraVariables);
    } else if (p == 2) {
        computeStructure<FastPower<2>>(*structure.getVolumes().getConservedVolume(),
        conservedVariables);
        computeStructure<FastPower<2>>(*structure.getVolumes().getExtraVolume(),
        extraVariables);
    } else if (p==3) {
        computeStructure<FastPower<3>>(*structure.getVolumes().getConservedVolume(),
        conservedVariables);
        computeStructure<FastPower<3>>(*structure.getVolumes().getExtraVolume(),
        extraVariables);
    } else if (p==4) {
        computeStructure<FastPower<4>>(*structure.getVolumes().getConservedVolume(),
        conservedVariables);
        computeStructure<FastPower<4>>(*structure.getVolumes().getExtraVolume(),
        extraVariables);
    } else  if (p ==5) {
        computeStructure<FastPower<5>>(*structure.getVolumes().getConservedVolume(),
        conservedVariables);
        computeStructure<FastPower<5>>(*structure.getVolumes().getExtraVolume(),
        extraVariables);
    } else {
        computeStructure<PowfPower>(*structure.getVolumes().getConservedVolume(),
        conservedVariables);
        computeStructure<PowfPower>(*structure.getVolumes().getExtraVolume(),
        extraVariables);
    }

}

void StructureCubeCUDA::finalizeStatistics() {

}

template<class PowerClass>
void StructureCubeCUDA::computeStructure(alsfvm::volume::Volume& output,
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

        structureOutput.resize(nx * ny * nz);
        const int dimensions = input.getDimensions();

        for (int h = 1; h < numberOfH; ++h) {
            const int threads = 1024;
            const int size = nx * ny * nz;
            const int blockNumber = (size + threads - 1) / threads;


            computeStructureCube<PowerClass> <<< blockNumber, threads>>>(thrust::raw_pointer_cast(
                    structureOutput.data()),
                inputView,
                h, nx, ny, nz, ngx, ngy, ngz, p, dimensions);


            real structureResult = thrust::reduce(structureOutput.begin(),
                    structureOutput.end(),
                    0.0, thrust::plus<real>());

            outputView.at(h) += structureResult / (nx * ny * nz);
        }

    }
}

REGISTER_STATISTICS(cuda, structure_cube, StructureCubeCUDA)
}
}
