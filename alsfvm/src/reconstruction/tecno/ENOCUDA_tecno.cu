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

#include "alsfvm/reconstruction/tecno/ENOCUDA.hpp"

#include "alsfvm/reconstruction/ENOCUDA.hpp"
#include "alsutils/error/Exception.hpp"
#include <cassert>
#include <cmath>
#include "alsfvm/reconstruction/ENOCoefficients.hpp"
#include <iostream>

#include <fstream>
#include "alsfvm/equation/equation_list.hpp"
#include "alsfvm/cuda/cuda_utils.hpp"


namespace alsfvm {
namespace reconstruction {
namespace tecno {



template<int order>
__global__ void performEnoReconstructionKernel(memory::View<const real>
    leftView,
    memory::View<const real> rightView,
    memory::View<real> left,
    memory::View<real> right,
    const gpu_array < real*, order - 1 > dividedDifferencesPointers,
    const gpu_array < gpu_array<real, order>, order + 1 > coefficients,
    int numberOfXCells,
    int numberOfYCells,
    int numberOfZCells,
    ivec3 directionVector
) {



    ivec3 coordinates = cuda::getCoordinates(threadIdx, blockIdx, blockDim,
            numberOfXCells,
            numberOfYCells,
            numberOfZCells,
            (order - 1) * directionVector);

    auto x = coordinates.x;
    auto y = coordinates.y;
    auto z = coordinates.z;

    if (x < 0 || y < 0 || z < 0) {
        return;
    }


    const size_t indexRight = leftView.index(x, y, z);


    const size_t indexLeft = rightView.index( (x - directionVector.x),
            (y - directionVector.y),
            (z - directionVector.z));

    // First we determine the shift
    // We do this by looping through the levels of the divided
    // differences, and each time we go left, we increment the shift.
    int shift = 0;

    for (int level = 0; level < order - 1; level++) {
        real dividedDifferenceRight = dividedDifferencesPointers[level][indexRight];
        real dividedDifferenceLeft = dividedDifferencesPointers[level][indexLeft];

        if (fabs(dividedDifferenceLeft) < fabs(dividedDifferenceRight)) {
            // Now we choose the left stencil
            shift++;
        }
    }

    // Now we have the stencil enabled. We need to find the correct
    // coefficients.

    auto coefficientsRight = coefficients[shift + 1];
    auto coefficientsLeft  = coefficients[shift];
    gpu_array < real, 2 * order - 1 > wCur;
    wCur[order - 1] = leftView.at(indexRight);

    for (int l = 0; l < order - 1; ++l) {
        wCur[order + l]   = wCur[order + l - 1] +
            dividedDifferencesPointers[0][indexRight + l];
        wCur[order - l - 2] = wCur[order - l - 1] -
            dividedDifferencesPointers[0][indexRight - l - 1];
    }

    // Calculate wlR, wrR
    real uli = 0, uri = 0;

    for (int l = 0; l < order; ++l) {
        uli += coefficientsLeft[l] * wCur[order - 1 - shift + l];
        uri += coefficientsRight[l] * wCur[order - 1 - shift + l];
    }

    left.at(indexRight) = uli;
    right.at(indexRight) = uri + (rightView.at(indexRight) - leftView.at(
                indexRight));
}

template<int order> ENOCUDA<order>::ENOCUDA(
    alsfvm::shared_ptr<memory::MemoryFactory>& memoryFactory,
    size_t nx, size_t ny, size_t nz)

    : memoryFactory(memoryFactory) {
    size_t ghostX = getNumberOfGhostCells();
    size_t ghostY = ny > 1 ? getNumberOfGhostCells() : 0;
    size_t ghostZ = nz > 1 ? getNumberOfGhostCells() : 0;
    makeDividedDifferenceArrays(nx + 2 * ghostX, ny + 2 * ghostY, nz + 2 * ghostZ);
}

template<int order>
void ENOCUDA<order>::performReconstruction(const volume::Volume& leftInput,
    const volume::Volume& rightInput,
    size_t direction,
    volume::Volume& leftOut,
    volume::Volume& rightOut) {
    // We often do compute order-1.
    static_assert(order > 0, "Can not do ENO reconstruction of order 0.");

    if (direction > 2) {
        THROW("Direction can only be 0, 1 or 2, was given: " << direction);
    }

    // Now we go on to do the actual reconstruction, choosing the stencil for
    // each point.
    const size_t nx = leftInput.getTotalNumberOfXCells();
    const size_t ny = leftInput.getTotalNumberOfYCells();
    const size_t nz = leftInput.getTotalNumberOfZCells();

    if (leftInput.getScalarMemoryArea(0)->getSize() !=
        dividedDifferences[0]->getSize()) {
        makeDividedDifferenceArrays(nx, ny, nz);
    }


    const ivec3 directionVector = make_direction_vector(direction);

    for (size_t var = 0; var < leftInput.getNumberOfVariables(); var++) {
        // make divided differences

        computeDividedDifferences(*leftInput.getScalarMemoryArea(var),
            *rightInput.getScalarMemoryArea(var),
            directionVector,
            1,
            *dividedDifferences[0]);

        for (size_t i = 1; i < order - 1; i++) {
            computeDividedDifferences(*dividedDifferences[i - 1],
                *dividedDifferences[i - 1],
                directionVector, i + 1, *dividedDifferences[i]);
        }

        // done computing divided differences



        // Sanity check, we need at least ONE point in the interior.
        assert(int(nx) > 2 * directionVector.x * order);
        assert((directionVector.y == 0u) || (int(ny) > 2 * directionVector.y * order));
        assert((directionVector.z == 0u) || (int(nz) > 2 * directionVector.z * order));



        gpu_array < real*, order - 1 > dividedDifferencesPointers;

        for (size_t i = 0; i < order - 1; i++) {
            dividedDifferencesPointers[i] = dividedDifferences[i]->getPointer();
        }

        gpu_array < gpu_array<real, order>, order + 1 > coefficients;

        for (size_t i = 0; i < order + 1; ++i) {
            for (size_t j = 0; j < order; ++j) {
                coefficients[i][j] = ENOCoeffiecients<order>::coefficients[i][j];
            }
        }

        const ivec3 start = (order - 1) * directionVector;
        const ivec3 end = ivec3(nx, ny, nz) - (order - 1) * directionVector;



        const int blockSize = 512;

        auto launchParameters = cuda::makeKernelLaunchParameters(start, end, blockSize);

        auto gridSize = std::get<0>(launchParameters);
        auto numberOfCellsPerDimension = std::get<1>(launchParameters);

        auto viewLeft = leftOut.getScalarMemoryArea(var)->getView();
        auto viewRight = rightOut.getScalarMemoryArea(var)->getView();
        auto viewInputLeft = leftInput.getScalarMemoryArea(var)->getView();
        auto viewInputRight = rightInput.getScalarMemoryArea(var)->getView();
#ifndef NDEBUG
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        CUDA_SAFE_CALL(cudaPeekAtLastError());
#endif
        performEnoReconstructionKernel<order> <<< gridSize, blockSize>>>(viewInputLeft,
            viewInputRight,
            viewLeft,
            viewRight,
            dividedDifferencesPointers,
            coefficients,
            numberOfCellsPerDimension.x,
            numberOfCellsPerDimension.y,
            numberOfCellsPerDimension.z,
            directionVector);
#ifndef NDEBUG
        CUDA_SAFE_CALL(cudaPeekAtLastError());
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
    }

    std::cout << "Reconstruction" << std::endl;
}

template<int order>
void ENOCUDA<order>::makeDividedDifferenceArrays(size_t nx, size_t ny,
    size_t nz) {

    for (size_t i = 0; i < dividedDifferences.size(); i++) {
        dividedDifferences[i] = memoryFactory->createScalarMemory(nx, ny, nz);
        dividedDifferences[i]->makeZero();
    }
}



template<int order>
size_t ENOCUDA<order>::getNumberOfGhostCells() const {
    return order;
}

__global__ void computeDividedDifferencesKernel(real* output,
    const real* inputLeft,
    const real* inputRight,
    size_t numberOfXCells, // total number of
    size_t numberOfYCells, // cells minus ghost cells
    size_t numberOfZCells, //
    size_t nx, // The total number of cells
    size_t ny, // in each
    size_t nz, // direction
    ivec3 direction,
    size_t level
) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;

    const size_t xInternalFormat = index % numberOfXCells;
    const size_t yInternalFormat = (index / numberOfXCells) % numberOfYCells;
    const size_t zInternalFormat = (index) / (numberOfXCells * numberOfYCells);

    if (xInternalFormat >= numberOfXCells || yInternalFormat >= numberOfYCells
        || zInternalFormat >= numberOfZCells) {
        return;
    }

    const int x = xInternalFormat + (level) * direction[0];
    const int y = yInternalFormat + (level) * direction[1];
    const int z = zInternalFormat + (level) * direction[2];


    const int indexRight = z * nx * ny + y * nx + x;
    const int indexLeft = (z - direction.z) * nx * ny
        + (y - direction.y) * nx
        + (x - direction.x);

    output[indexLeft] = inputLeft[indexRight] - inputRight[indexLeft];

}

template<int order>
void ENOCUDA<order>::computeDividedDifferences(const memory::Memory<real>&
    inputLeft,
    const memory::Memory<real>& inputRight,
    const ivec3& direction,
    size_t level,
    memory::Memory<real>& output) {


    const int nx = inputLeft.getSizeX();
    const int ny = inputLeft.getSizeY();
    const int nz = inputLeft.getSizeZ();

    // Sanity check, we need at least ONE point in the interior.
    assert(nx > int(2 * direction.x * level));
    assert(ny > int(2 * direction.y * level));
    assert(nz > int(2 * direction.z * level));

    const ivec3 start = int(level) * direction;
    const ivec3 end = ivec3(nx, ny, nz) - int(level) * direction;

    const real* pointerInLeft = inputLeft.getPointer();
    const real* pointerInRight = inputRight.getPointer();

    real* pointerOut = output.getPointer();


    const size_t blockSize = 1024;

    const ivec3 numberOfCellsPerDimension = end - start;

    const size_t totalNumberOfCells = size_t(numberOfCellsPerDimension.x) *
        size_t(numberOfCellsPerDimension.y) *
        size_t(numberOfCellsPerDimension.z);

    const size_t gridSize = (totalNumberOfCells + blockSize - 1 ) / blockSize;

    computeDividedDifferencesKernel <<< gridSize, blockSize>>>(pointerOut,
        pointerInLeft, pointerInRight,
        numberOfCellsPerDimension.x,
        numberOfCellsPerDimension.y,
        numberOfCellsPerDimension.z,
        nx, ny, nz,
        direction,
        level);
}

template class ENOCUDA<2>;
template class ENOCUDA<3>;
template class ENOCUDA<4>;

}
}
}
