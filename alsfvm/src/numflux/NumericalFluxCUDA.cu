#include "alsfvm/numflux/NumericalFluxCUDA.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/numflux/euler/HLL.hpp"
#include "alsfvm/numflux/euler/HLL3.hpp"
#include <array>
#include <iostream>
#include "alsfvm/cuda/cuda_utils.hpp"

namespace alsfvm { namespace numflux { 

	namespace {
		template<class Flux, class Equation, size_t dimension, bool xDir, bool yDir, bool zDir, size_t direction>
		__global__ void computeFluxDevice(typename Equation::ConstViews left, typename Equation::ConstViews right, typename Equation::Views output,
			size_t numberOfXCells, size_t numberOfYCells, size_t numberOfZCells, const real cellScaling, size_t numberOfGhostCells) {

			const size_t index = threadIdx.x + blockDim.x * blockIdx.x;
			// We have
			// index = z * nx * ny + y * nx + x;
			const size_t xInternalFormat = index % numberOfXCells; 
			const size_t yInternalFormat = (index / numberOfYCells) % numberOfYCells;
			const size_t zInternalFormat = (index) / (numberOfXCells * numberOfYCells);

			const size_t x = xInternalFormat +  numberOfGhostCells;
			const size_t y = yInternalFormat +  numberOfGhostCells;
			const size_t z = zInternalFormat +  numberOfGhostCells;

			if (xInternalFormat >= numberOfXCells || yInternalFormat >= numberOfYCells || zInternalFormat >= numberOfZCells) {
				return;
			}

			const size_t leftIndex = output.index(x - xDir, y - yDir, z - zDir);
			const size_t rightIndex = output.index(x + yDir, y + yDir, z + zDir);
			const size_t middleIndex = output.index(x, y, z);

			typename Equation::AllVariables leftJpHf = Equation::fetchAllVariables(right, middleIndex);
			typename Equation::AllVariables rightJpHf = Equation::fetchAllVariables(left, rightIndex);

			typename Equation::AllVariables leftJmHf = Equation::fetchAllVariables(right, leftIndex);
			typename Equation::AllVariables rightJmHf = Equation::fetchAllVariables(left, middleIndex);

			// F(U_j, U_r)
			typename Equation::ConservedVariables fluxMiddleRight;
			Flux::template computeFlux<direction>(leftJpHf, rightJpHf, fluxMiddleRight);

			typename Equation::ConservedVariables fluxLeftMiddle;
			Flux::template computeFlux<direction>(leftJmHf, rightJmHf, fluxLeftMiddle);

			Equation::addToViewAt(output, middleIndex, cellScaling*(fluxLeftMiddle)-cellScaling*(fluxMiddleRight));
		}

		template<class Flux, class Equation,  size_t dimension, bool xDir, bool yDir, bool zDir, size_t direction>
		void computeFlux(const volume::Volume& left, const volume::Volume& right, volume::Volume& output, size_t numberOfGhostCells, real cellScaling) {

			const size_t numberOfXCells = left.getNumberOfXCells() - 2 * numberOfGhostCells;
			const size_t numberOfYCells = left.getNumberOfYCells() - 2 * (dimension > 1) * numberOfGhostCells;
			const size_t numberOfZCells = left.getNumberOfZCells() - 2 * (dimension > 2) * numberOfGhostCells;

			typename Equation::ConstViews viewLeft(left);
			typename Equation::ConstViews viewRight(right);
			typename Equation::Views viewOut(output);

			size_t totalSize = numberOfXCells * numberOfYCells * numberOfZCells;
			size_t blockSize = 1024;
			computeFluxDevice <Flux, Equation, dimension, xDir, yDir, zDir, direction>
				<< <(totalSize + blockSize - 1) /blockSize, blockSize>> >
				(viewLeft, viewRight, viewOut, numberOfXCells, numberOfYCells, numberOfZCells, cellScaling, numberOfGhostCells);
			
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
			CUDA_SAFE_CALL(cudaGetLastError());
		}

		template<class Flux, class Equation, size_t dimension>
		void callComputeFlux(const volume::Volume& left, const volume::Volume& right, volume::Volume& output, size_t numberOfGhostCells, rvec3 cellScaling) {
			computeFlux<Flux, Equation, dimension, 1, 0, 0, 0>(left, right, output, numberOfGhostCells, cellScaling.x);

			if (dimension > 1) {
				computeFlux<Flux, Equation, dimension, 0, 1, 0, 1>(left, right, output, numberOfGhostCells, cellScaling.y);
			} 
			if (dimension > 2) {
				computeFlux<Flux, Equation, dimension, 0, 0, 1, 2>(left, right, output, numberOfGhostCells, cellScaling.z);
			}
		}


	}

	template<class Flux, class Equation, size_t dimension>
	NumericalFluxCUDA<Flux, Equation, dimension>::NumericalFluxCUDA(const grid::Grid &grid,
		std::shared_ptr<reconstruction::Reconstruction>& reconstruction,
		std::shared_ptr<DeviceConfiguration> &deviceConfiguration)
		: reconstruction(reconstruction)
	{
		static_assert(dimension > 0, "We only support positive dimension!");
		static_assert(dimension < 4, "We only support dimension up to 3");

		std::shared_ptr<memory::MemoryFactory> memoryFactory(new memory::MemoryFactory(deviceConfiguration));
		volume::VolumeFactory volumeFactory(Equation::name, memoryFactory);

		left = volumeFactory.createConservedVolume(grid.getDimensions().x,
			grid.getDimensions().y,
			grid.getDimensions().z,
			getNumberOfGhostCells());
		left->makeZero();

		right = volumeFactory.createConservedVolume(grid.getDimensions().x,
			grid.getDimensions().y,
			grid.getDimensions().z,
			getNumberOfGhostCells());

		right->makeZero();
	}

	template<class Flux, class Equation, size_t dimension>
	void NumericalFluxCUDA<Flux, Equation, dimension>::computeFlux(const volume::Volume& conservedVariables,
		const rvec3& cellLengths,
		volume::Volume& output
		)
	{
		static_assert(dimension > 0, "We only support positive dimension!");
		static_assert(dimension < 4, "We only support dimension up to 3");

		output.makeZero();

		callComputeFlux<Flux, Equation, dimension>(conservedVariables, conservedVariables, output, getNumberOfGhostCells(), cellLengths);
	}

	/// 
	/// \returns the number of ghost cells this specific flux requires
	///
	template<class Flux, class Equation, size_t dimension>
	size_t NumericalFluxCUDA<Flux, Equation, dimension>::getNumberOfGhostCells() {
		return reconstruction->getNumberOfGhostCells();
	}

	template class NumericalFluxCUDA < euler::HLL, equation::euler::Euler, 1 >;
	template class NumericalFluxCUDA < euler::HLL, equation::euler::Euler, 2 >;
	template class NumericalFluxCUDA < euler::HLL, equation::euler::Euler, 3 >;

	template class NumericalFluxCUDA < euler::HLL3, equation::euler::Euler, 1 >;
	template class NumericalFluxCUDA < euler::HLL3, equation::euler::Euler, 2 >;
	template class NumericalFluxCUDA < euler::HLL3, equation::euler::Euler, 3 >;
}
}
