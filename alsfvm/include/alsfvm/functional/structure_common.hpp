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

#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/memory/Memory.hpp"
#include "alsfvm/volume/Volume.hpp"
#include "alsutils/math/FastPower.hpp"
#include "alsutils/math/PowPower.hpp"
#include "alsfvm/boundary/ValueAtBoundary.hpp"

namespace alsfvm {
namespace functional {

inline __device__ __host__ int makePositive(int position, int N) {
    if (position < 0) {
        position += N;
    }

    return position;

}


template<alsfvm::boundary::Type BoundaryType, class Function>
__device__ __host__ void forEachPointInComputeStructureCube(
    Function f,
    const alsfvm::memory::View<const real>& input,
    int i, int j, int k, int h, int nx, int ny, int nz,
    int ngx, int ngy, int ngz, int dimensions) {
    const auto u = input.at(i + ngx, j + ngy, k + ngz);

    const auto numberOfCellsWithoutGhostCells = ivec3{nx, ny, nz};
    const auto numberOfGhostCells = ivec3{ngx, ngy, ngz};

    for (int d = 0; d < dimensions; d++) {
        // side = 0 represents bottom, side = 1 represents top
        for (int side = 0; side < 2; side++) {
            const bool zDir = (d == 2);
            const bool yDir = (d == 1);
            const bool xDir = (d == 0);
            // Either we start on the left (i == 0), or on the right(i==1)
            const int zStart = zDir ?
                (side == 0 ? k - h : k + h) : (dimensions > 2 ? k - h : 0);

            const int zEnd = zDir ?
                (zStart + 1) : (dimensions > 2 ? k + h + 1 : 1);

            int yStart = yDir ?
                (side == 0 ? j - h : j + h) : (dimensions == 2 ? j - h + 1 :
                    (dimensions == 3 ? j - h + 1 : 0));

            int yEnd = yDir ?
                (yStart + 1) : (dimensions == 2 ? j + h : (dimensions == 3 ? j + h : 1));

            int xStart = xDir ?
                (side == 0 ? i - h : i + h) : i - h;

            int xEnd = xDir ?
                (xStart + 1) : i + h + 1;

            if (zDir) {
                xStart += 1;
                xEnd -= 1;

                //yStart += 1;
                //yEnd -= 1;
            }

            for (int z = zStart; z < zEnd; z++) {
                for (int y = yStart; y < yEnd; y++) {
                    for (int x = xStart; x < xEnd; x++) {
                        const auto discretePosition = ivec3{i, j, k};
                        const auto discretePositionPlusH = ivec3{x, y, z};


                        const auto u_ijk_h =
                            alsfvm::boundary::ValueAtBoundary<BoundaryType>::getValueAtBoundary(
                                input,
                                discretePositionPlusH,
                                numberOfCellsWithoutGhostCells,
                                numberOfGhostCells);


                        f(u, u_ijk_h);
                    }
                }
            }
        }
    }
}

template<alsfvm::boundary::Type BoundaryType, class PowerClass>
__device__ __host__ void computeStructureCube(
    alsfvm::memory::View<real>&
    output,
    const alsfvm::memory::View<const real>& input,
    int i, int j, int k, int h, int nx, int ny, int nz,
    int ngx, int ngy, int ngz, int dimensions, real p) {

    forEachPointInComputeStructureCube<BoundaryType>([&](double u, double u_h) {
        output.at(h) += PowerClass::power(fabs(u - u_h), p) / (nx * ny * nz);
    }, input, i, j, k, h, nx, ny, nz, ngx, ngy, ngz, dimensions);
}




template<alsfvm::boundary::Type BoundaryType, class PowerClass>
inline void computeStructureCubeCPU(alsfvm::volume::Volume& output,
    const alsfvm::volume::Volume& input, int numberOfH, double p) {
    for (size_t var = 0; var < input.getNumberOfVariables(); ++var) {
        auto inputView = input[var]->getView();
        auto outputView = output[var]->getView();

        int ngx = int(input.getNumberOfXGhostCells());
        int ngy = int(input.getNumberOfYGhostCells());
        int ngz = int(input.getNumberOfZGhostCells());

        int nx = int(input.getNumberOfXCells());
        int ny = int(input.getNumberOfYCells());
        int nz = int(input.getNumberOfZCells());

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    for (int h = 1; h < numberOfH; ++h) {

                        computeStructureCube<BoundaryType, PowerClass>(outputView, inputView, i, j, k,
                            h, nx, ny, nz,
                            ngx, ngy, ngz, input.getDimensions(), p);

                    }
                }
            }
        }


    }


}

template<alsfvm::boundary::Type BoundaryType>
inline void dispatchComputeStructureCubeCPU(alsfvm::volume::Volume& output,
    const alsfvm::volume::Volume& input, int numberOfH, double p) {
    if (p == 1.0) {
        computeStructureCubeCPU <BoundaryType, alsutils::math::FastPower<1>>
            (output, input, numberOfH, p);
    } else if (p == 2.0) {
        computeStructureCubeCPU <BoundaryType, alsutils::math::FastPower<2>>
            (output, input, numberOfH, p);
    }

    else if (p == 3.0) {
        computeStructureCubeCPU <BoundaryType, alsutils::math::FastPower<3>>
            (output, input, numberOfH, p);
    } else if (p == 4.0) {
        computeStructureCubeCPU <BoundaryType, alsutils::math::FastPower<4>>
            (output, input, numberOfH, p);
    } else if (p == 5.0) {
        computeStructureCubeCPU <BoundaryType, alsutils::math::FastPower<5>>
            (output, input, numberOfH, p);
    } else {
        computeStructureCubeCPU <BoundaryType, alsutils::math::PowPower>
        (output, input, numberOfH, p);
    }
}

}
}
