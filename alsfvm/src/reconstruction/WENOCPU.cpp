#include "alsfvm/reconstruction/WENOCPU.hpp"
#include <array>

namespace alsfvm { namespace reconstruction { 

	template<int order>
	WENOCPU<order>::WENOCPU() {
		//empty
	}

	template<int order>
	void WENOCPU<order>::performReconstruction(const volume::Volume& inputVariables,
		size_t direction,
		size_t indicatorVariable,
		volume::Volume& leftOut,
		volume::Volume& rightOut) 
	{
		// We often do compute order-1.
		static_assert(order > 0, "Can not do WENO reconstruction of order 0.");

		if (direction > 2) {
			THROW("Direction can only be 0, 1 or 2, was given: " << direction);
		}
		const ivec3 directionVector(direction == 0, direction == 1, direction == 2);

		

		// Now we go on to do the actual reconstruction, choosing the stencil for
		// each point.
		const size_t nx = inputVariables.getTotalNumberOfXCells();
		const size_t ny = inputVariables.getTotalNumberOfYCells();
		const size_t nz = inputVariables.getTotalNumberOfZCells();

		// Sanity check, we need at least ONE point in the interior.
		assert(nx > 2 * directionVector.x * order);
		assert((directionVector.y == 0) || (ny > 2 * directionVector.y * order));
		assert((directionVector.z == 0) || (nz > 2 * directionVector.z * order));

		const size_t startX = directionVector.x * order;
		const size_t startY = directionVector.y * order;
		const size_t startZ = directionVector.z * order;

		const size_t endX = nx - directionVector.x * order;
		const size_t endY = ny - directionVector.y * order;
		const size_t endZ = nz - directionVector.z * order;


		
		std::vector<const real*> pointers(inputVariables.getNumberOfVariables());

		for (size_t var = 0; var < inputVariables.getNumberOfVariables(); var++) {
			pointers[var] = inputVariables.getScalarMemoryArea(var)->getPointer();
		}

		const real* pointerIn = inputVariables.getScalarMemoryArea(var)->getPointer();
		real* pointerOutLeft = leftOut.getScalarMemoryArea(var)->getPointer();
		real* pointerOutRight = rightOut.getScalarMemoryArea(var)->getPointer();

		for (size_t z = startZ; z < endZ; z++) {
			for (size_t y = startY; y < endY; y++) {
				for (size_t x = startX; x < endX; x++) {

					std::array<real, order> beta(0);
					const size_t indexRight = z*nx*ny + y * nx + x;
					const size_t indexLeft = (z - directionVector.z) * nx * ny
						+ (y - directionVector.y) * nx
						+ (x - directionVector.x);

					for (int shift = 0; shift < order; shift++) {


						auto coefficientsRight = ENOCoeffiecients<order>::coefficients[shift + 1];
						auto coefficientsLeft = ENOCoeffiecients<order>::coefficients[shift];
						real leftValue = 0.0;
						real rightValue = 0.0;
						for (size_t j = 0; j < order; j++) {

							const size_t index = (z - (shift - j)*directionVector.z) * nx * ny
								+ (y - (shift - j)*directionVector.y) * nx
								+ (x - (shift - j)*directionVector.x);


							const real value = pointerIn[index];
							leftValue += coefficientsLeft[j] * value;
							rightValue += coefficientsRight[j] * value;
						}


					}

					pointerOutLeft[indexRight] = leftValue;
					pointerOutRight[indexRight] = rightValue;
					assert(!std::isnan(leftValue));
					assert(!std::isnan(rightValue));


				}
			}
		}



	}

	///
	/// \brief getNumberOfGhostCells returns the number of ghost cells we need
	///        for this computation
	/// \return order.
	///
	template<int order>
	virtual size_t WENOCPU<order>::getNumberOfGhostCells() {
		return order + 1;
	}
}
}
