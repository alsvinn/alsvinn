#include "alsfvm/reconstruction/WENO2CUDA.hpp"
#include "alsfvm/reconstruction/WENOCoefficients.hpp"
#include "alsfvm/reconstruction/ENOCoefficients.hpp"
#include "alsfvm/cuda/cuda_utils.hpp"
#include "alsfvm/equation/euler/Euler.hpp"



namespace alsfvm {
	namespace reconstruction {
		namespace {


			template<class Equation, size_t dimension, bool xDir, bool yDir, bool zDir>
			__global__ void weno2Device(typename Equation::ConstViews input,
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
				const size_t x = xInternalFormat + (1) * xDir;
				const size_t y = yInternalFormat + (1) * yDir;
				const size_t z = zInternalFormat + (1) * zDir;

				const size_t indexOut = left.index(x, y, z);
				const size_t indexRight = left.index(x + xDir, y + yDir, z + zDir);
				const size_t indexLeft = left.index(x - xDir, y - yDir, z - zDir);
				const real i0 = pointerInWeight[indexOut];
				const real b0 = square(pointerInWeight[indexRight] - i0);
				const real b1 = square(i0 - pointerInWeight[indexLeft]);
				const real a0 = 1 / (3 * (ALSFVM_WENO_EPSILON + b0)*(ALSFVM_WENO_EPSILON + b0));
				const real a1 = 2 / (3 * (ALSFVM_WENO_EPSILON + b1)*(ALSFVM_WENO_EPSILON + b1));
				const real w0 = a0 / (a0 + a1);
				const real w1 = a1 / (a0 + a1);

				
				for (int var = 0; var < Equation::getNumberOfConservedVariables(); var++) {
					memory::View<const real> in = input.get(var);
					left.get(var).at(indexOut) =
					  0.5*(w1 * in.at(indexLeft) +
						(3 * w0 + w1) * in.at(indexOut) -
						w0 * in.at(indexRight));
					right.get(var).at(indexOut) = 0.5*(w0 * in.at(indexRight) +
						(3 * w1 + w0) * in.at(indexOut) -
						w1 * in.at(indexLeft));

				}


			}

			template<class Equation, size_t dimension, bool xDir, bool yDir, bool zDir >
			void callReconstructionDevice(const volume::Volume& inputVariables,
				size_t direction,
				size_t indicatorVariable,
				volume::Volume& leftOut,
				volume::Volume& rightOut) {

				const size_t numberOfXCells = leftOut.getTotalNumberOfXCells() - 2 * (1) * xDir;
				const size_t numberOfYCells = leftOut.getTotalNumberOfYCells() - 2 * (1) * yDir;
				const size_t numberOfZCells = leftOut.getTotalNumberOfZCells() - 2 * (1) * zDir;

				const size_t totalSize = numberOfXCells * numberOfYCells * numberOfZCells;


				const size_t blockSize = 512;
				const size_t gridSize = (totalSize + blockSize - 1) / blockSize;



				typename Equation::Views viewLeft(leftOut);
				typename Equation::Views viewRight(rightOut);
				typename Equation::ConstViews viewInput(inputVariables);

				weno2Device<Equation, dimension, xDir, yDir, zDir> << <gridSize, blockSize >> >(viewInput,
					viewLeft, viewRight, inputVariables.getScalarMemoryArea(indicatorVariable)->getPointer(),
					numberOfXCells, numberOfYCells, numberOfZCells);

			}

			template<size_t dimension, class Equation>
			void performReconstructionDevice(const volume::Volume& inputVariables,
				size_t direction,
				size_t indicatorVariable,
				volume::Volume& leftOut,
				volume::Volume& rightOut) {
				assert(direction < 3);
				switch (direction) {
				case 0:
					callReconstructionDevice<Equation, dimension, 1, 0, 0>(inputVariables, direction, indicatorVariable, leftOut, rightOut);
					break;

				case 1:
					callReconstructionDevice<Equation, dimension, 0, 1, 0>(inputVariables, direction, indicatorVariable, leftOut, rightOut);
					break;

				case 2:
					callReconstructionDevice<Equation, dimension, 0, 0, 1>(inputVariables, direction, indicatorVariable, leftOut, rightOut);
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
		template<class Equation>
		void WENO2CUDA<Equation>::performReconstruction(const volume::Volume& inputVariables,
			size_t direction,
			size_t indicatorVariable,
			volume::Volume& leftOut,
			volume::Volume& rightOut)
		{
			size_t dimension = 1 + (leftOut.getNumberOfYCells() > 1) + (leftOut.getNumberOfZCells() > 1);

			switch (dimension) {
			case 1:
				performReconstructionDevice<1, Equation>(inputVariables, direction, indicatorVariable, leftOut, rightOut);
				break;
			case 2:
				performReconstructionDevice<2, Equation>(inputVariables, direction, indicatorVariable, leftOut, rightOut);
				break;
			case 3:
				performReconstructionDevice<3, Equation>(inputVariables, direction, indicatorVariable, leftOut, rightOut);
				break;
			}

		}

		template<class Equation>
		WENO2CUDA<Equation>::WENO2CUDA() {
			

		}

		template class WENO2CUDA < equation::euler::Euler>;
		template class WENO2CUDA < equation::euler::Euler>;

	}
}
