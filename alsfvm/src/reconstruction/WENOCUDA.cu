#include "alsfvm/reconstruction/WENOCUDA.hpp"
#include <boost/array.hpp>
#define NUMBER_OF_ENO_COEFFICIENTS_PER_ORDER 5
__constant__ alsfvm::real enoCoefficients[NUMBER_OF_ENO_COEFFICIENTS_PER_ORDER * 4];

namespace alsfvm { namespace reconstruction {
	namespace {
		template<size_t order, class Equation, bool xDir, bool yDir, bool zDir>
		__global__ void wenoDevice(typename Equation::ConstViews input,
			typename Equation::Views left, typename Equation::Views right,
			real* pointerInWeight,
			size_t numberOfXCells, size_t numberOfYCells, size_t numberOfZCells) {

			const size_t index = threadIdx.x + blockDim.x * blockIdx.x;
			// We have
			// index = z * nx * ny + y * nx + x;
			const size_t xInternalFormat = index % numberOfXCells;
			const size_t yInternalFormat = (index / numberOfXCells) % numberOfYCells;
			const size_t zInternalFormat = (index) / (numberOfXCells * numberOfYCells);

			const size_t x = xInternalFormat + order;
			const size_t y = yInternalFormat + (dimension > 1) * (order);
			const size_t z = zInternalFormat + (dimension > 2) * (order);

			// First we need to find alpha and beta.
			real stencil[2 * order - 1];
			for (int i = -order; i < order - 1; i++) {
				const size_t index = input.index((z + i * zDir) ,
					(y + i*yDir),
					(x + i*zDir));

				stencil[i + order] = pointerInWeight[index];
			}

			real alphaRight[order];
			real alphaLeft[order];
			real alphaRightSum = 0.0;
			real alphaLeftSum = 0.0;

			computeAlpha<int(order) - 1, int(order)>(stencil,
				alphaLeftSum,
				alphaRightSum,
				alphaLeft,
				alphaRight);

			real* coefficients = enoCoefficients + NUMBER_OF_ENO_COEFFICIENTS_PER_ORDER * (order - 2);

			for (int var = 0; var < Equation::numberOfConservedVariables; var++) {
				real leftWenoValue = 0.0;
				real rightWenoValue = 0.0;
				// Loop through all stencils (shift = r)
				for (int shift = 0; shift < order; shift++) {

					real* coefficientsRight = coefficients + shift + 1;
					real* coefficientsLeft = coefficients + shift;
					real leftValue = 0.0;
					real rightValue = 0.0;
					for (size_t j = 0; j < order; j++) {

						const size_t index = left.index( (z - (shift - j)*directionVector.z),
							(y - (shift - j)*directionVector.y),
							(x - (shift - j)*directionVector.x));


						const real value = input.get(var).at(index);
						leftValue += coefficientsLeft[j] * value;
						rightValue += coefficientsRight[j] * value;
					}
					leftWenoValue += leftValue * alphaLeft[shift] / alphaLeftSum;
					rightWenoValue += rightValue * alphaRight[shift] / alphaRightSum;
				}

				left.get(var).at(indexRight) = leftWenoValue;
				right.get(var).at(indexRight) = rightWenoValue;

			}


		}
	}
	///
	/// Performs reconstruction.
	/// \param[in] inputVariables the variables to reconstruct.
	/// \param[in] direction the direction:
	/// direction | explanation
	/// ----------|------------
	///     0     |   x-direction
	///     1     |   y-direction
	///     2     |   z-direction
	///
	/// \param[in] indicatorVariable the variable number to use for
	/// stencil selection. We will determine the stencil based on
	/// inputVariables->getScalarMemoryArea(indicatorVariable).
	///
	/// \param[out] leftOut at the end, will contain the left interpolated values
	///                     for all grid cells in the interior.
	///
	/// \param[out] rightOut at the end, will contain the right interpolated values
	///                     for all grid cells in the interior.
	///
	template<size_t order>
	void WENOCUDA<order>::performReconstruction(const volume::Volume& inputVariables,
		size_t direction,
		size_t indicatorVariable,
		volume::Volume& leftOut,
		volume::Volume& rightOut) 
	{


	}


}
}
