#include "alsfvm/reconstruction/WENOCUDA.hpp"
#include "alsfvm/reconstruction/WENOCoefficients.hpp"
#include "alsfvm/reconstruction/ENOCoefficients.hpp"
#include "alsfvm/cuda/cuda_utils.hpp"
#include "alsfvm/equation/euler/Euler.hpp"

#define NUMBER_OF_ENO_LEVELS 5
#define NUMBER_OF_ENO_COEFFICIENTS_PER_ORDER 5*5
#define NUMBER_OF_WENO_LEVELS 4
#define NUMBER_OF_WENO_COEFFICIENTS_PER_ORDER 5

__constant__ alsfvm::real enoCoefficients[NUMBER_OF_ENO_COEFFICIENTS_PER_ORDER * NUMBER_OF_ENO_LEVELS];
__constant__ alsfvm::real wenoCoefficients[NUMBER_OF_WENO_COEFFICIENTS_PER_ORDER * NUMBER_OF_WENO_LEVELS];

namespace alsfvm { namespace reconstruction {
	namespace {
		template<int order>
		__device__ real wenoCoefficient(int i) {
			return wenoCoefficients[NUMBER_OF_WENO_COEFFICIENTS_PER_ORDER * order + i];
		}

		template<int order, int i>
		struct Alpha {
			template<class S, class T>
			static  __device__ void computeAlpha(S& stencil,
				real& sumLeft, real& sumRight,
				T& left,
							     T& right) {

			  const real epsilon = ALSFVM_WENO_EPSILON;
			  const real beta = WENOCoefficients<order>::template computeBeta<i>(stencil);
			  right[i] = wenoCoefficient<order>(i) / pow(beta + epsilon, 2);
			  sumRight += right[i];

			  left[i] = wenoCoefficient<order>(order - 1 - i) / pow(beta + epsilon, 2);
			  sumLeft += left[i];

			  Alpha<order, i - 1>::computeAlpha(stencil, sumLeft, sumRight, left, right);

			}
		};

		template<int order>
		struct Alpha <order,  -1 > {
			template<class S, class T>
			static  __device__ void computeAlpha(S& stencil,
				real& sumLeft, real& sumRight,
				T& left,
				T& right) {
				// empty
			}
		};


		template<size_t order, class Equation, size_t dimension, bool xDir, bool yDir, bool zDir>
		__global__ void wenoDevice(typename Equation::ConstViews input,
			typename Equation::Views left, typename Equation::Views right,
			const real* pointerInWeight,
			size_t numberOfXCells, size_t numberOfYCells, size_t numberOfZCells) {

			const size_t index = threadIdx.x + blockDim.x * blockIdx.x;
			// We have
			// index = z * nx * ny + y * nx + x;
			const size_t xInternalFormat = index % numberOfXCells;
			const size_t yInternalFormat = (index / numberOfXCells) % numberOfYCells;
			const size_t zInternalFormat = (index) / (numberOfXCells * numberOfYCells);

			if (xInternalFormat >= numberOfXCells || yInternalFormat >= numberOfYCells || zInternalFormat >= numberOfZCells) {
			  return;
			}
			const size_t x = xInternalFormat + (order - 1) * xDir;
			const size_t y = yInternalFormat + (order - 1) * yDir;
			const size_t z = zInternalFormat + (order - 1) * zDir;

			const size_t indexOut = left.index(x, y, z);

			// First we need to find alpha and beta.
			real stencil[2 * order - 1];
			for (int i = -order + 1; i < order; i++) {
				const size_t indexIn = input.index((x + i * xDir) ,
					(y + i*yDir),
					(z + i*zDir));

				stencil[i + order - 1] = pointerInWeight[indexIn];
			}

			real alphaRight[order];
			real alphaLeft[order];
			real alphaRightSum = 0.0;
			real alphaLeftSum = 0.0;

			Alpha<int(order), int(order) - 1>::computeAlpha(stencil,
				alphaLeftSum,
				alphaRightSum,
				alphaLeft,
				alphaRight);

			const real* coefficients = enoCoefficients + NUMBER_OF_ENO_COEFFICIENTS_PER_ORDER * (order);

			for (int var = 0; var < Equation::getNumberOfConservedVariables(); var++) {
				real leftWenoValue = 0.0;
				real rightWenoValue = 0.0;
				// Loop through all stencils (shift = r)
				for (int shift = 0; shift < order; shift++) {

				    const real* coefficientsRight = coefficients + (shift + 1) * order;
					const real* coefficientsLeft = coefficients + shift * order;
					real leftValue = 0.0;
					real rightValue = 0.0;
					for (size_t j = 0; j < order; j++) {

						const size_t inputIndex = left.index( (x - (shift - j)*xDir),
							(y - (shift - j)*yDir),
							(z - (shift - j)*zDir));


						const real value = input.get(var).at(inputIndex);
						leftValue += coefficientsLeft[j] * value;
						rightValue += coefficientsRight[j] * value;
					}
					leftWenoValue += leftValue * alphaLeft[shift] / alphaLeftSum;
					rightWenoValue += rightValue * alphaRight[shift] / alphaRightSum;
				}

				left.get(var).at(indexOut) = leftWenoValue;
				right.get(var).at(indexOut) = rightWenoValue;

			}


		}

		template<class Equation, size_t order, size_t dimension, bool xDir, bool yDir, bool zDir >
		void callReconstructionDevice(const volume::Volume& inputVariables,
			size_t direction,
			size_t indicatorVariable,
			volume::Volume& leftOut,
			volume::Volume& rightOut) {

			const size_t numberOfXCells = leftOut.getTotalNumberOfXCells() - 2 * (order - 1) * xDir;
			const size_t numberOfYCells = leftOut.getTotalNumberOfYCells() - 2 * (order - 1) * yDir;
			const size_t numberOfZCells = leftOut.getTotalNumberOfZCells() - 2 * (order - 1) * zDir;

			const size_t totalSize = numberOfXCells * numberOfYCells * numberOfZCells;


			const size_t blockSize = 512;
			const size_t gridSize = (totalSize + blockSize - 1) / blockSize;



			typename Equation::Views viewLeft(leftOut);
			typename Equation::Views viewRight(rightOut);
			typename Equation::ConstViews viewInput(inputVariables);

			wenoDevice<order, Equation, dimension, xDir, yDir, zDir> << <gridSize, blockSize >> >(viewInput,
				viewLeft, viewRight, inputVariables.getScalarMemoryArea(indicatorVariable)->getPointer(),
				numberOfXCells, numberOfYCells, numberOfZCells);

		}

		template<size_t dimension, class Equation, size_t order>
		void performReconstructionDevice(const volume::Volume& inputVariables,
			size_t direction,
			size_t indicatorVariable,
			volume::Volume& leftOut,
			volume::Volume& rightOut) {
			assert(direction < 3);
			switch (direction) {
			case 0:
				callReconstructionDevice<Equation, order, dimension, 1, 0, 0>(inputVariables, direction, indicatorVariable, leftOut, rightOut);
				break;

			case 1:
				callReconstructionDevice<Equation, order, dimension, 0, 1, 0>(inputVariables, direction, indicatorVariable, leftOut, rightOut);
				break;

			case 2:
				callReconstructionDevice<Equation, order, dimension, 0, 0, 1>(inputVariables, direction, indicatorVariable, leftOut, rightOut);
				break;
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
	template<class Equation, size_t order>
	void WENOCUDA<Equation, order>::performReconstruction(const volume::Volume& inputVariables,
		size_t direction,
		size_t indicatorVariable,
		volume::Volume& leftOut,
		volume::Volume& rightOut) 
	{
		size_t dimension = 1 + (leftOut.getNumberOfYCells() > 1) + (leftOut.getNumberOfZCells() > 1);

		switch (dimension) {
		case 1:
			performReconstructionDevice<1, Equation, order>(inputVariables, direction, indicatorVariable, leftOut, rightOut);
			break;
		case 2:
			performReconstructionDevice<2, Equation, order>(inputVariables, direction, indicatorVariable, leftOut, rightOut);
			break;
		case 3:
			performReconstructionDevice<3, Equation, order>(inputVariables, direction, indicatorVariable, leftOut, rightOut);
			break;
		}

	}

	template<class Equation, size_t order>
	WENOCUDA<Equation, order>::WENOCUDA() {
		std::vector<real> enoCoefficientsHost(NUMBER_OF_ENO_LEVELS * NUMBER_OF_ENO_COEFFICIENTS_PER_ORDER, 0);
		for (size_t shift = 0; shift < order + 1; ++shift) {
			for (size_t i = 0; i < order; ++i) {
			  enoCoefficientsHost[order * NUMBER_OF_ENO_COEFFICIENTS_PER_ORDER + shift*order + i] = ENOCoeffiecients<order>::coefficients[shift][i];
			}
		}

		CUDA_SAFE_CALL(cudaMemcpyToSymbol(enoCoefficients, enoCoefficientsHost.data(), enoCoefficientsHost.size() * sizeof(real)));


		std::vector<real> wenoCoefficientsHost(NUMBER_OF_WENO_LEVELS * NUMBER_OF_WENO_COEFFICIENTS_PER_ORDER, 0);
		for (size_t i = 0; i < order; ++i) {
				wenoCoefficientsHost[order * NUMBER_OF_WENO_COEFFICIENTS_PER_ORDER + i] = WENOCoefficients<order>::coefficients[i];
		}

		CUDA_SAFE_CALL(cudaMemcpyToSymbol(wenoCoefficients, wenoCoefficientsHost.data(), wenoCoefficientsHost.size() * sizeof(real)));

	}

	template class WENOCUDA < equation::euler::Euler, 2 > ;
	template class WENOCUDA < equation::euler::Euler, 3 >;

}
}
