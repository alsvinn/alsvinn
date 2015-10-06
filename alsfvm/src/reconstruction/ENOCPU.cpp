#include "alsfvm/reconstruction/ENOCPU.hpp"
#include "alsfvm/error/Exception.hpp"
#include <cassert>
#include <cmath>
#include "alsfvm/reconstruction/ENOCoefficients.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include <iostream>

#include <fstream>

namespace alsfvm { namespace reconstruction {

template<int order>
ENOCPU<order>::ENOCPU(alsfvm::shared_ptr<memory::MemoryFactory> &memoryFactory,
                      size_t nx, size_t ny, size_t nz)
{
	size_t ghostX = getNumberOfGhostCells();
	size_t ghostY = ny > 1 ? getNumberOfGhostCells() : 0;
	size_t ghostZ = nz > 1 ? getNumberOfGhostCells() : 0;
    for(size_t i = 0; i < dividedDifferences.size(); i++) {
        dividedDifferences[i] = memoryFactory->createScalarMemory(nx + 2*ghostX, ny + 2*ghostY, nz + 2*ghostZ);
        dividedDifferences[i]->makeZero();
    }
}

template<int order>
void ENOCPU<order>::performReconstruction(const volume::Volume &inputVariables,
	size_t direction,
	size_t indicatorVariable,
	volume::Volume &leftOut,
	volume::Volume &rightOut)
{
	// We often do compute order-1.
	static_assert(order > 0, "Can not do ENO reconstruction of order 0.");

	if (direction > 2) {
		THROW("Direction can only be 0, 1 or 2, was given: " << direction);
	}
	const ivec3 directionVector(direction == 0, direction == 1, direction == 2);

	// make divided differences
	computeDividedDifferences(*inputVariables.getScalarMemoryArea(indicatorVariable),
		directionVector,
		1,
		*dividedDifferences[0]);

	for (size_t i = 1; i < order - 1; i++) {
		computeDividedDifferences(*dividedDifferences[i - 1],
			directionVector, i + 1, *dividedDifferences[i]);
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

	const size_t startX = directionVector.x * (order - 1);
	const size_t startY = directionVector.y * (order - 1);
	const size_t startZ = directionVector.z * (order - 1);

	const size_t endX = nx - directionVector.x * (order - 1);
	const size_t endY = ny - directionVector.y * (order - 1);
	const size_t endZ = nz - directionVector.z * (order - 1);

	std::array<real*, order - 1> dividedDifferencesPointers;

	for (size_t i = 0; i < order - 1; i++) {
		dividedDifferencesPointers[i] = dividedDifferences[i]->getPointer();
	}

	for (size_t var = 0; var < inputVariables.getNumberOfVariables(); var++) {
		const real* pointerIn = inputVariables.getScalarMemoryArea(var)->getPointer();
		real* pointerOutLeft = leftOut.getScalarMemoryArea(var)->getPointer();
		real* pointerOutRight = rightOut.getScalarMemoryArea(var)->getPointer();

		for (size_t z = startZ; z < endZ; z++) {
			for (size_t y = startY; y < endY; y++) {
#pragma omp parallel for
				for (size_t x = startX; x < endX; x++) {


					const size_t indexRight = z*nx*ny + y * nx + x;
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

						if (std::abs(dividedDifferenceLeft) < std::abs(dividedDifferenceRight)) {
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

						const size_t index = (z - (shift - j)*directionVector.z) * nx * ny
							+ (y - (shift - j)*directionVector.y) * nx
							+ (x - (shift - j)*directionVector.x);


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
size_t ENOCPU<order>::getNumberOfGhostCells()
{
    return order;
}

template<int order>
void ENOCPU<order>::computeDividedDifferences(const memory::Memory<real>& input,
                                              const ivec3& direction,
                                              size_t level,
                                              memory::Memory<real>& output)
{


    const size_t nx = input.getSizeX();
    const size_t ny = input.getSizeY();
    const size_t nz = input.getSizeZ();

    // Sanity check, we need at least ONE point in the interior.
    assert(nx > 2*direction.x * level);
    assert(ny > 2*direction.y * level);
    assert(nz > 2*direction.z * level);

    const size_t startX = direction.x * level;
    const size_t startY = direction.y * level;
    const size_t startZ = direction.z * level;

    const size_t endX = nx - direction.x * level;
    const size_t endY = ny - direction.y * level;
    const size_t endZ = nz - direction.z * level;

    const real* pointerIn = input.getPointer();

    real* pointerOut = output.getPointer();
	
    for (size_t z = startZ; z < endZ; z++) {
        for(size_t y = startY; y < endY; y++) {
            for(size_t x = startX; x < endX; x++) {
                const size_t indexRight = z*nx*ny + y * nx + x;
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
