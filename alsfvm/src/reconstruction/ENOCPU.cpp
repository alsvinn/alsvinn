#include "alsfvm/reconstruction/ENOCPU.hpp"
#include "alsutils/error/Exception.hpp"
#include <cassert>
#include <cmath>
#include "alsfvm/reconstruction/ENOCoefficients.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include <iostream>

#include <fstream>
#include "alsutils/log.hpp"

namespace alsfvm {
namespace reconstruction {

template<int order> ENOCPU<order>::ENOCPU(
    alsfvm::shared_ptr<memory::MemoryFactory>& memoryFactory,
    size_t nx, size_t ny, size_t nz)
    : memoryFactory(memoryFactory) {
    size_t ghostX = getNumberOfGhostCells();
    size_t ghostY = ny > 1 ? getNumberOfGhostCells() : 0;
    size_t ghostZ = nz > 1 ? getNumberOfGhostCells() : 0;
    makeDividedDifferenceArrays(nx + 2 * ghostX, ny + 2 * ghostY, nz + 2 * ghostZ);
}

template<int order>
void ENOCPU<order>::performReconstruction(const volume::Volume& inputVariables,
    size_t direction,
    size_t indicatorVariable,
    volume::Volume& leftOut,
    volume::Volume& rightOut, const ivec3& start, const ivec3& end) {
    // We often do compute order-1.
    static_assert(order > 0, "Can not do ENO reconstruction of order 0.");
    const int nx = inputVariables.getTotalNumberOfXCells();
    const int ny = inputVariables.getTotalNumberOfYCells();
    const int nz = inputVariables.getTotalNumberOfZCells();

    if (direction > 2) {
        THROW("Direction can only be 0, 1 or 2, was given: " << direction);
    }

    if (inputVariables.getScalarMemoryArea(0)->getSize() !=
        dividedDifferences[0]->getSize()) {
        makeDividedDifferenceArrays(nx, ny, nz);
    }

    const ivec3 directionVector(direction == 0, direction == 1, direction == 2);

    // make divided differences
    computeDividedDifferences(*inputVariables.getScalarMemoryArea(
            indicatorVariable),
        directionVector,
        1,
        *dividedDifferences[0],
        start,
        end);

    for (size_t i = 1; i < order - 1; i++) {
        computeDividedDifferences(*dividedDifferences[i - 1],
            directionVector, i + 1, *dividedDifferences[i],
            start,
            end);
    }

    // done computing divided differences

    // Now we go on to do the actual reconstruction, choosing the stencil for
    // each point.


    // Sanity check, we need at least ONE point in the interior.
    assert(int(nx) > 2 * directionVector.x * order);
    assert((directionVector.y == 0u) || (int(ny) > 2 * directionVector.y * order));
    assert((directionVector.z == 0u) || (int(nz) > 2 * directionVector.z * order));

    const int startX =           (order - directionVector.x * 1) + start.x;
    const int startY =  (ny > 1) * (order - directionVector.y * 1) + start.y;
    const int startZ =  (nz > 1) * (order - directionVector.z * 1) + start.z;

    const int endX = nx -          (order - directionVector.x) + end.x;
    const int endY = ny - (ny > 1) * (order - directionVector.y) + end.y;
    const int endZ = nz - (nz > 1) * (order - directionVector.z) + end.z;

    std::array < real*, order - 1 > dividedDifferencesPointers;

    for (size_t i = 0; i < order - 1; i++) {
        dividedDifferencesPointers[i] = dividedDifferences[i]->getPointer();
    }

#if 0
    static std::array<int, 3> first;

    if (first[direction] < 8) {
        ALSVINN_LOG(INFO, "new " << direction << " start = (" << startX << ", " <<
            startY << ", " << startZ << ")");
        ALSVINN_LOG(INFO, "new " << direction << "end   = (" << endX << ", " << endY <<
            ", " << endZ << ")");
    }

    first[direction]++;
#endif

    for (size_t var = 0; var < inputVariables.getNumberOfVariables(); var++) {
        const real* pointerIn = inputVariables.getScalarMemoryArea(var)->getPointer();
        real* pointerOutLeft = leftOut.getScalarMemoryArea(var)->getPointer();
        real* pointerOutRight = rightOut.getScalarMemoryArea(var)->getPointer();

        for (int z = startZ; z < endZ; z++) {
            #pragma omp parallel for

            for (int y = startY; y < endY; y++) {
                #pragma omp simd

                for (int x = startX; x < int(endX); x++) {


                    const size_t indexRight = z * nx * ny + y * nx + x;
                    const size_t indexLeft = (z - directionVector.z) * nx * ny
                        + (y - directionVector.y) * nx
                        + (x - directionVector.x);

                    // First we determine the shift
                    // We do this by looping through the levels of the divided
                    // differences, and each time we go left, we increment the shift.
                    int shift = 0;

                    for (size_t level = 0; level < order - 1; level++) {
                        real dividedDifferenceRight = dividedDifferencesPointers[level][indexRight];
                        real dividedDifferenceLeft = dividedDifferencesPointers[level][indexLeft];

                        if (std::fabs(dividedDifferenceLeft) < std::fabs(dividedDifferenceRight)) {
                            // Now we choose the left stencil
                            shift++;
                        }


                    }

                    // Now we have the stencil enabled. We need to find the correct
                    // coefficients.

                    auto coefficientsRight = ENOCoeffiecients<order>::coefficients[shift + 1];
                    auto coefficientsLeft = ENOCoeffiecients<order>::coefficients[shift];
                    real leftValue = 0.0;
                    real rightValue = 0.0;

                    for (int j = 0; j < order; j++) {

                        const size_t index = (z - (shift - j) * directionVector.z) * nx * ny
                            + (y - (shift - j) * directionVector.y) * nx
                            + (x - (shift - j) * directionVector.x);


                        const real value = pointerIn[index];
                        leftValue += coefficientsLeft[j] * value;
                        rightValue += coefficientsRight[j] * value;

                    }

                    pointerOutLeft[indexRight] = leftValue;
                    pointerOutRight[indexRight] = rightValue;
                    assert(!std::isnan(leftValue));
                    assert(!std::isnan(rightValue));


                }
            }
        }
    }

}


template<int order>
size_t ENOCPU<order>::getNumberOfGhostCells() {
    return order;
}

template<int order>
void ENOCPU<order>::makeDividedDifferenceArrays(size_t nx, size_t ny,
    size_t nz) {

    for (size_t i = 0; i < dividedDifferences.size(); i++) {
        dividedDifferences[i] = memoryFactory->createScalarMemory(nx, ny, nz);
        dividedDifferences[i]->makeZero();
    }
}

template<int order>
void ENOCPU<order>::computeDividedDifferences(const memory::Memory<real>& input,
    const ivec3& direction,
    size_t level,
    memory::Memory<real>& output,
    const ivec3& start,
    const ivec3& end) {


    const int nx = input.getSizeX();
    const int ny = input.getSizeY();
    const int nz = input.getSizeZ();


    // Sanity check, we need at least ONE point in the interior.
    assert(nx > 2 * direction.x * level);
    assert(ny > 2 * direction.y * level);
    assert(nz > 2 * direction.z * level);

    const int startX =        (direction.x == 0) * order + direction.x * level +
        start.x;
    const int startY = (ny > 1) * (direction.y == 0) * order + direction.y * level +
        start.y;
    const int startZ = (nz > 1) * (direction.z == 0) * order + direction.z * level +
        start.z;

    const int endX = nx -        ((direction.x == 0) * (order - 1) + direction.x *
            (level - 1)) + end.x;
    const int endY = ny - ((ny > 1) * (direction.y == 0) * (order - 1) +
            direction.y * (level - 1)) + end.y;
    const int endZ = nz - ((nz > 1) * (direction.z == 0) * (order - 1) +
            direction.z * (level - 1)) + end.z;

#if 0
    static std::array<int, order * 3> first;

    if (first[(direction[1])*order + level] < 8) {
        ALSVINN_LOG(INFO, "level = " << level << " direction = " << direction <<
            " start = (" << startX << ", " << startY << ", " << startZ << ")");
        ALSVINN_LOG(INFO, "level = " << level << " direction = " << direction <<
            " end   = (" << endX << ", " << endY << ", " << endZ << ")");
    }

    first[(direction[1])*order + level]++;
#endif
    const real* pointerIn = input.getPointer();

    real* pointerOut = output.getPointer();

    for (int z = startZ; z < endZ; z++) {
        for (int y = startY; y < endY; y++) {
            for (int x = startX; x < endX; x++) {
                const size_t indexRight = z * nx * ny + y * nx + x;
                const size_t indexLeft = (z - direction.z) * nx * ny
                    + (y - direction.y) * nx
                    + (x - direction.x);

                pointerOut[indexLeft] = pointerIn[indexRight] - pointerIn[indexLeft];

            }
        }
    }
}

template class ENOCPU<1>;
template class ENOCPU<2>;
template class ENOCPU<3>;
template class ENOCPU<4>;



}
}
