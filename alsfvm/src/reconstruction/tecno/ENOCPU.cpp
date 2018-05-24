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

#include "alsfvm/reconstruction/tecno/ENOCPU.hpp"
#include "alsfvm/reconstruction/ENOCoefficients.hpp"

namespace alsfvm {
namespace reconstruction {
namespace tecno {

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
void ENOCPU<order>::performReconstruction(const volume::Volume& leftInput,
    const volume::Volume& rightInput,
    size_t direction,
    volume::Volume& leftOut,
    volume::Volume& rightOut) {

    // We often do compute order-1.
    static_assert(order > 0, "Can not do ENO reconstruction of order 0.");
    const size_t nx = leftInput.getTotalNumberOfXCells();
    const size_t ny = leftInput.getTotalNumberOfYCells();
    const size_t nz = leftInput.getTotalNumberOfZCells();

    if (direction > 2) {
        THROW("Direction can only be 0, 1 or 2, was given: " << direction);
    }

    if (leftInput.getScalarMemoryArea(0)->getSize() !=
        dividedDifferences[0]->getSize()) {
        makeDividedDifferenceArrays(nx, ny, nz);
    }

    const ivec3 directionVector(direction == 0, direction == 1, direction == 2);



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

        // Now we go on to do the actual reconstruction, choosing the stencil for
        // each point.


        // Sanity check, we need at least ONE point in the interior.
        assert(int(nx) > 2 * directionVector.x * order);
        assert((directionVector.y == 0u) || (int(ny) > 2 * directionVector.y * order));
        assert((directionVector.z == 0u) || (int(nz) > 2 * directionVector.z * order));

        const size_t startX = directionVector.x * (order - 1);
        const size_t startY = directionVector.y * (order - 1);
        const size_t startZ = directionVector.z * (order - 1);

        const size_t endX = nx - directionVector.x * (order - 1);
        const size_t endY = ny - directionVector.y * (order - 1);
        const size_t endZ = nz - directionVector.z * (order - 1);

        std::array < real*, order - 1 > dividedDifferencesPointers;

        for (size_t i = 0; i < order - 1; i++) {
            dividedDifferencesPointers[i] = dividedDifferences[i]->getPointer();
        }



        auto leftView = leftInput.getScalarMemoryArea(var)->getView();
        auto rightView = rightInput.getScalarMemoryArea(var)->getView();


        real* pointerOutLeft = leftOut.getScalarMemoryArea(var)->getPointer();
        real* pointerOutRight = rightOut.getScalarMemoryArea(var)->getPointer();

        for (size_t z = startZ; z < endZ; z++) {
            #pragma omp parallel for

            for (size_t y = startY; y < endY; y++) {
                #pragma omp simd

                for (int x = startX; x < int(endX); x++) {



                    const size_t indexRight =  leftView.index(x, y, z);
                    const size_t indexLeft = leftView.index((x - directionVector.x),
                            (y - directionVector.y),
                            (z - directionVector.z));

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

                    std::array < real, 2 * order - 1 > wCur;
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

                    pointerOutLeft[indexRight] = uli;
                    pointerOutRight[indexRight] = uri + (rightView.at(indexRight) - leftView.at(
                                indexRight));
                    assert(!std::isnan(uli));
                    assert(!std::isnan(uri));


                }
            }
        }
    }

}


template<int order>
size_t ENOCPU<order>::getNumberOfGhostCells() const {
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
void ENOCPU<order>::computeDividedDifferences(const memory::Memory<real>&
    leftInput,
    const memory::Memory<real>& rightInput,
    const ivec3& direction,
    size_t level,
    memory::Memory<real>& output) {


    const size_t nx = leftInput.getSizeX();
    const size_t ny = leftInput.getSizeY();
    const size_t nz = leftInput.getSizeZ();


    // Sanity check, we need at least ONE point in the interior.
    assert(nx > 2 * direction.x * level);
    assert(ny > 2 * direction.y * level);
    assert(nz > 2 * direction.z * level);

    const size_t startX = direction.x * level;
    const size_t startY = direction.y * level;
    const size_t startZ = direction.z * level;

    const size_t endX = nx - direction.x * level;
    const size_t endY = ny - direction.y * level;
    const size_t endZ = nz - direction.z * level;

    auto leftView = leftInput.getView();
    auto rightView = rightInput.getView();

    real* pointerOut = output.getPointer();


    for (size_t z = startZ; z < endZ; z++) {
        for (size_t y = startY; y < endY; y++) {
            #pragma omp parallel for

            for (size_t x = startX; x < endX; x++) {
                const size_t indexRight = z * nx * ny + y * nx + x;
                const size_t indexLeft = (z - direction.z) * nx * ny
                    + (y - direction.y) * nx
                    + (x - direction.x);

                pointerOut[indexLeft] = leftView.at(indexRight) - rightView.at(indexLeft);

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
}
