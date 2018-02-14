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



template<class Equation, int order>
__global__ void performEnoReconstructionKernel(typename Equation::ConstViews
    input,
    typename Equation::Views left,
    typename Equation::Views right,
    const gpu_array < real*, order - 1 > dividedDifferencesPointers,
    const gpu_array < gpu_array<real, order>, order + 1 > coefficients,
    int numberOfXCells,
    int numberOfYCells,
    int numberOfZCells,
    ivec3 directionVector,
    ivec3 start,
    ivec3 spaceFillingVector
) {



    ivec3 coordinates = cuda::getCoordinates(threadIdx, blockIdx, blockDim,
            numberOfXCells,
            numberOfYCells,
            numberOfZCells,
            (order) * spaceFillingVector - directionVector + start);

    int x = coordinates.x;
    int y = coordinates.y;
    int z = coordinates.z;

    if (x < 0 || y < 0 || z < 0) {
        return;
    }




    const size_t indexRight = input.index(x, y, z);


    const size_t indexLeft = input.index( (x - directionVector.x),
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
    typename Equation::ConservedVariables leftValue;
    typename Equation::ConservedVariables rightValue;

    for (int j = 0; j < order; j++) {

        const size_t index = input.index((x - (shift - j) * directionVector.x),
                (y - (shift - j) * directionVector.y),
                (z - (shift - j) * directionVector.z));


        const auto value = Equation::fetchConservedVariables(input, index);
        leftValue = leftValue + coefficientsLeft[j] * value;
        rightValue = rightValue + coefficientsRight[j] * value;

    }

    Equation::setViewAt(left, indexRight, leftValue);
    Equation::setViewAt(right, indexRight, rightValue);
}

template<class Equation, int order> ENOCUDA<Equation, order>::ENOCUDA(
    alsfvm::shared_ptr<memory::MemoryFactory>& memoryFactory,
    size_t nx, size_t ny, size_t nz) {
    size_t ghostX = getNumberOfGhostCells();
    size_t ghostY = ny > 1 ? getNumberOfGhostCells() : 0;
    size_t ghostZ = nz > 1 ? getNumberOfGhostCells() : 0;

    for (size_t i = 0; i < dividedDifferences.size(); i++) {
        dividedDifferences[i] = memoryFactory->createScalarMemory(nx + 2 * ghostX,
                ny + 2 * ghostY, nz + 2 * ghostZ);
        dividedDifferences[i]->makeZero();
    }
}

template<class Equation, int order>
void ENOCUDA<Equation, order>::performReconstruction(const volume::Volume&
    inputVariables,
    size_t direction,
    size_t indicatorVariable,
    volume::Volume& leftOut,
    volume::Volume& rightOut,
    const ivec3& startIndex,
    const ivec3& endIndex) {
    // We often do compute order-1.
    static_assert(order > 0, "Can not do ENO reconstruction of order 0.");

    if (direction > 2) {
        THROW("Direction can only be 0, 1 or 2, was given: " << direction);
    }

    const ivec3 directionVector = make_direction_vector(direction);


    // make divided differences

    computeDividedDifferences(*inputVariables.getScalarMemoryArea(
            indicatorVariable),
        directionVector,
        1,
        *dividedDifferences[0], startIndex, endIndex);

    for (size_t i = 1; i < order - 1; i++) {
        computeDividedDifferences(*dividedDifferences[i - 1],
            directionVector, i + 1, *dividedDifferences[i], startIndex, endIndex);
    }

    // done computing divided differences

    // Now we go on to do the actual reconstruction, choosing the stencil for
    // each point.
    const size_t nx = inputVariables.getTotalNumberOfXCells();
    const size_t ny = inputVariables.getTotalNumberOfYCells();
    const size_t nz = inputVariables.getTotalNumberOfZCells();

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

    const ivec3 spaceFillingVector = make_space_filling_vector(nx, ny, nz);
    const ivec3 start = (order * spaceFillingVector - directionVector) + startIndex;
    const ivec3 end = ivec3(nx, ny,
            nz) - (order * spaceFillingVector - directionVector) + endIndex;


    const int blockSize = 512;

    auto launchParameters = cuda::makeKernelLaunchParameters(start, end, blockSize);

    auto gridSize = std::get<0>(launchParameters);
    auto numberOfCellsPerDimension = std::get<1>(launchParameters);

    typename Equation::Views viewLeft(leftOut);
    typename Equation::Views viewRight(rightOut);
    typename Equation::ConstViews viewInput(inputVariables);

#ifndef NDEBUG
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaPeekAtLastError());
#endif
    performEnoReconstructionKernel<Equation, order> <<< gridSize, blockSize>>>
    (viewInput,
        viewLeft,
        viewRight,
        dividedDifferencesPointers,
        coefficients,
        numberOfCellsPerDimension.x,
        numberOfCellsPerDimension.y,
        numberOfCellsPerDimension.z,
        directionVector, startIndex,
        spaceFillingVector);
#ifndef NDEBUG
    CUDA_SAFE_CALL(cudaPeekAtLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
}


template<class Equation, int order>
size_t ENOCUDA<Equation, order>::getNumberOfGhostCells() {
    return order;
}

template<int order>
__global__ void computeDividedDifferencesKernel(real* output,
    const real* input,
    size_t numberOfXCells, // total number of
    size_t numberOfYCells, // cells minus ghost cells
    size_t numberOfZCells, //
    size_t nx, // The total number of cells
    size_t ny, // in each
    size_t nz, // direction
    ivec3 direction,
    size_t level,
    ivec3 start,
    ivec3 spaceFillingVector
) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;

    const size_t xInternalFormat = index % numberOfXCells;
    const size_t yInternalFormat = (index / numberOfXCells) % numberOfYCells;
    const size_t zInternalFormat = (index) / (numberOfXCells * numberOfYCells);

    if (xInternalFormat >= numberOfXCells || yInternalFormat >= numberOfYCells
        || zInternalFormat >= numberOfZCells) {
        return;
    }

    const int x = xInternalFormat + (level) * direction[0] +
        (order ) * spaceFillingVector.x * (direction[0] == 0) + start.x;
    const int y = yInternalFormat + (level) * direction[1] +
        (order ) * spaceFillingVector.y * (direction[1] == 0) + start.y;
    const int z = zInternalFormat + (level) * direction[2] +
        (order ) * spaceFillingVector.z * (direction[2] == 0) + start.z;

    const int indexRight = z * nx * ny + y * nx + x;
    const int indexLeft = (z - direction.z) * nx * ny
        + (y - direction.y) * nx
        + (x - direction.x);

    output[indexLeft] = input[indexRight] - input[indexLeft];

}

template<class Equation, int order>
void ENOCUDA<Equation, order>::computeDividedDifferences(
    const memory::Memory<real>& input,
    const ivec3& direction,
    size_t level,
    memory::Memory<real>& output,
    const ivec3& startIndex,
    const ivec3& endIndex) {


    const int nx = input.getSizeX();
    const int ny = input.getSizeY();
    const int nz = input.getSizeZ();

    const auto spaceFillingVector = make_space_filling_vector(nx, ny, nz);

    // Sanity check, we need at least ONE point in the interior.
    assert(nx > int(2 * direction.x * level));
    assert(ny > int(2 * direction.y * level));
    assert(nz > int(2 * direction.z * level));

    const ivec3 directionComplement = ivec3(direction.x == 0, direction.y == 0,
            direction.z == 0);
    const ivec3 start = int(level) * direction + (order) * directionComplement *
        spaceFillingVector + startIndex;
    const ivec3 end = ivec3(nx, ny,
            nz) - int(level - 1) * direction - (order - 1) * directionComplement *
        spaceFillingVector + endIndex;

    const real* pointerIn = input.getPointer();

    real* pointerOut = output.getPointer();


    const size_t blockSize = 1024;

    const ivec3 numberOfCellsPerDimension = end - start;

    const size_t totalNumberOfCells = size_t(numberOfCellsPerDimension.x) *
        size_t(numberOfCellsPerDimension.y) *
        size_t(numberOfCellsPerDimension.z);

    const size_t gridSize = (totalNumberOfCells + blockSize - 1 ) / blockSize;

    computeDividedDifferencesKernel<order> <<< gridSize, blockSize>>>(pointerOut,
        pointerIn,
        numberOfCellsPerDimension.x,
        numberOfCellsPerDimension.y,
        numberOfCellsPerDimension.z,
        nx, ny, nz,
        direction,
        level,
        startIndex,
        spaceFillingVector);
}

template class ENOCUDA<alsfvm::equation::euler::Euler<1>, 2>;
template class ENOCUDA<alsfvm::equation::euler::Euler<1>, 3>;
template class ENOCUDA<alsfvm::equation::euler::Euler<1>, 4>;

template class ENOCUDA<alsfvm::equation::euler::Euler<2>, 2>;
template class ENOCUDA<alsfvm::equation::euler::Euler<2>, 3>;
template class ENOCUDA<alsfvm::equation::euler::Euler<2>, 4>;

template class ENOCUDA<alsfvm::equation::euler::Euler<3>, 2>;
template class ENOCUDA<alsfvm::equation::euler::Euler<3>, 3>;
template class ENOCUDA<alsfvm::equation::euler::Euler<3>, 4>;

template class ENOCUDA<alsfvm::equation::burgers::Burgers, 2>;
template class ENOCUDA<alsfvm::equation::burgers::Burgers, 3>;
template class ENOCUDA<alsfvm::equation::burgers::Burgers, 4>;

}
}
