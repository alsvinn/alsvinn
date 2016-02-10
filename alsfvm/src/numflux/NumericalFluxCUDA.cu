#include "alsfvm/numflux/NumericalFluxCUDA.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/numflux/euler/HLL.hpp"
#include "alsfvm/numflux/euler/HLL3.hpp"
#include <thrust/device_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <iostream>
#include "alsfvm/cuda/cuda_utils.hpp"

namespace alsfvm { namespace numflux { 

	namespace {

		template<class Equation, size_t dimension, bool xDir, bool yDir, bool zDir, size_t direction>
		__global__ void combineFluxDevice(Equation equation, typename Equation::ConstViews input, typename Equation::Views output,
			const size_t numberOfXCells, const size_t numberOfYCells, const size_t numberOfZCells, const size_t numberOfGhostCells) {

			const size_t index = threadIdx.x + blockDim.x * blockIdx.x;


			const size_t xInternalFormat = index % numberOfXCells;
			const size_t yInternalFormat = (index / numberOfXCells) % numberOfYCells;
			const size_t zInternalFormat = (index) / (numberOfXCells * numberOfYCells);

			const size_t x = xInternalFormat + numberOfGhostCells - xDir;
			const size_t y = yInternalFormat + (dimension > 1) * numberOfGhostCells - yDir;
			const size_t z = zInternalFormat + (dimension > 2) * numberOfGhostCells - zDir;

			if (xInternalFormat >= numberOfXCells || yInternalFormat >= numberOfYCells || zInternalFormat >= numberOfZCells) {
				return;
			}

			const size_t rightIndex = output.index(x + xDir, y + yDir, z + zDir);
			const size_t middleIndex = output.index(x, y, z);

			typename Equation::ConservedVariables fluxMiddle = (-1.0)* equation.fetchConservedVariables(input, middleIndex);
			typename Equation::ConservedVariables fluxRight = equation.fetchConservedVariables(input, rightIndex);
			equation.addToViewAt(output, rightIndex, (fluxMiddle + fluxRight));
		}

		template<class Flux, class Equation, size_t dimension, bool xDir, bool yDir, bool zDir, size_t direction>
		__global__ void computeFluxDevice(Equation equation, typename Equation::ConstViews left, typename Equation::ConstViews right, typename Equation::Views output,
			const size_t numberOfXCells, const size_t numberOfYCells, const size_t numberOfZCells, real* waveSpeeds, const size_t numberOfGhostCells) {
			const size_t index = threadIdx.x + blockDim.x * blockIdx.x;
			// We have
			// index = z * nx * ny + y * nx + x;
			const size_t xInternalFormat = index % numberOfXCells; 
			const size_t yInternalFormat = (index / numberOfXCells) % numberOfYCells;
			const size_t zInternalFormat = (index) / (numberOfXCells * numberOfYCells);
			
			const size_t x = xInternalFormat +  numberOfGhostCells - xDir;
			const size_t y = yInternalFormat + (dimension > 1) * (numberOfGhostCells - yDir);
			const size_t z = zInternalFormat + (dimension > 2) * (numberOfGhostCells - zDir);

			if (xInternalFormat >= numberOfXCells || yInternalFormat >= numberOfYCells || zInternalFormat >= numberOfZCells) {
				return;
			}

			

			//const size_t leftIndex = output.index(x - xDir, y - yDir, z - zDir);
			const size_t rightIndex = output.index(x + xDir, y + yDir, z + zDir);
			const size_t middleIndex = output.index(x, y, z);

			typename Equation::AllVariables leftJpHf =  equation.fetchAllVariables(right, middleIndex);
			typename Equation::AllVariables rightJpHf = equation.fetchAllVariables(left, rightIndex);


			typename Equation::ConservedVariables fluxMiddleRight;
			waveSpeeds[middleIndex] = Flux::template computeFlux<direction>(equation, leftJpHf, rightJpHf, fluxMiddleRight);

			
			
			equation.setViewAt(output, middleIndex, (-1.0)*fluxMiddleRight);

			

		}

		template<class Flux, class Equation,  size_t dimension, bool xDir, bool yDir, bool zDir, size_t direction>
        void computeFlux(const Equation& equation, const volume::Volume& left, const volume::Volume& right, volume::Volume& output, size_t numberOfGhostCells, real& waveSpeed) {
			static thrust::device_vector<real> waveSpeeds;
			waveSpeeds.resize(left.getScalarMemoryArea(0)->getSize(), 0.0);
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
			CUDA_SAFE_CALL(cudaGetLastError());

			const size_t numberOfXCells = left.getTotalNumberOfXCells() - 2 * numberOfGhostCells + xDir;
			const size_t numberOfYCells = left.getTotalNumberOfYCells() - 2 * (dimension > 1) * numberOfGhostCells + yDir;
			const size_t numberOfZCells = left.getTotalNumberOfZCells() - 2 * (dimension > 2) * numberOfGhostCells + zDir;
			
			typename Equation::ConstViews viewLeft(left);
			typename Equation::ConstViews viewRight(right);
			typename Equation::Views viewOut(output);

			size_t totalSize = numberOfXCells * numberOfYCells * numberOfZCells;

			size_t blockSize = 128;
			computeFluxDevice <Flux, Equation, dimension, xDir, yDir, zDir, direction>
				<< <(totalSize + blockSize - 1) /blockSize, blockSize>> >
				(equation, viewLeft, viewRight, viewOut, numberOfXCells, numberOfYCells, numberOfZCells, thrust::raw_pointer_cast(&waveSpeeds[0]), numberOfGhostCells);
			
			waveSpeed = thrust::reduce(waveSpeeds.begin(), waveSpeeds.end(), 0.0, thrust::maximum<real>());
			CUDA_SAFE_CALL(cudaDeviceSynchronize());

			
			CUDA_SAFE_CALL(cudaGetLastError());
		}


		template<class Equation, size_t dimension, bool xDir, bool yDir, bool zDir, size_t direction>
		void combineFlux(const Equation& equation, const volume::Volume& input, volume::Volume& output, size_t numberOfGhostCells) {
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
			CUDA_SAFE_CALL(cudaGetLastError());

			const size_t numberOfXCells = input.getTotalNumberOfXCells() - 2 * numberOfGhostCells;
			const size_t numberOfYCells = input.getTotalNumberOfYCells() - 2 * (dimension > 1) * numberOfGhostCells;
			const size_t numberOfZCells = input.getTotalNumberOfZCells() - 2 * (dimension > 2) * numberOfGhostCells;

			typename Equation::ConstViews inputView(input);
			typename Equation::Views viewOut(output);

			size_t totalSize = numberOfXCells * numberOfYCells * numberOfZCells;

			size_t blockSize = 128;
			combineFluxDevice <Equation, dimension, xDir, yDir, zDir, direction>
				<< <(totalSize + blockSize - 1) / blockSize, blockSize >> >
				(equation, inputView, viewOut, numberOfXCells, numberOfYCells, numberOfZCells, numberOfGhostCells);

			CUDA_SAFE_CALL(cudaDeviceSynchronize());
			CUDA_SAFE_CALL(cudaGetLastError());
		}

		template<class Flux, class Equation, size_t dimension>
		void callComputeFlux(const Equation& equation, const volume::Volume& conservedVariables, volume::Volume& left, volume::Volume& right, volume::Volume& output, volume::Volume& temporaryOutput, size_t numberOfGhostCells, rvec3& waveSpeeds,
			reconstruction::Reconstruction& reconstruction) {
			reconstruction.performReconstruction(conservedVariables, 0, 0, left, right);
            computeFlux<Flux, Equation, dimension, 1, 0, 0, 0>(equation, left, right, temporaryOutput, numberOfGhostCells, waveSpeeds.x);
            combineFlux<Equation, dimension, 1, 0, 0, 0>(equation, temporaryOutput, output, numberOfGhostCells);

			if (dimension > 1) {
				reconstruction.performReconstruction(conservedVariables, 1, 0, left, right);
                computeFlux<Flux, Equation, dimension, 0, 1, 0, 1>(equation, left, right, temporaryOutput, numberOfGhostCells, waveSpeeds.y);
                combineFlux<Equation, dimension, 0, 1, 0, 1>(equation, temporaryOutput, output, numberOfGhostCells);
			} 
			if (dimension > 2) {
				reconstruction.performReconstruction(conservedVariables, 2, 0, left, right);
                computeFlux<Flux, Equation, dimension, 0, 0, 1, 2>(equation, left, right, temporaryOutput, numberOfGhostCells, waveSpeeds.z);
                combineFlux<Equation, dimension, 0, 0, 1, 2>(equation, temporaryOutput, output, numberOfGhostCells);
			}

		}

	}

	template<class Flux, class Equation, size_t dimension>
	NumericalFluxCUDA<Flux, Equation, dimension>::NumericalFluxCUDA(const grid::Grid &grid,
		alsfvm::shared_ptr<reconstruction::Reconstruction>& reconstruction,
        simulator::SimulatorParameters& parameters,
		alsfvm::shared_ptr<DeviceConfiguration> &deviceConfiguration)
        : reconstruction(reconstruction), equation(static_cast<typename Equation::Parameters&>(parameters.getEquationParameters()))
	{
		static_assert(dimension > 0, "We only support positive dimension!");
		static_assert(dimension < 4, "We only support dimension up to 3");

		alsfvm::shared_ptr<memory::MemoryFactory> memoryFactory(new memory::MemoryFactory(deviceConfiguration));
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

		fluxOutput = volumeFactory.createConservedVolume(grid.getDimensions().x,
			grid.getDimensions().y,
			grid.getDimensions().z,
			getNumberOfGhostCells());
	}

	template<class Flux, class Equation, size_t dimension>
	void NumericalFluxCUDA<Flux, Equation, dimension>::computeFlux(const volume::Volume& conservedVariables,
		rvec3& waveSpeeds, bool computeWaveSpeeds, 
		volume::Volume& output
		)
	{

		static_assert(dimension > 0, "We only support positive dimension!");
		static_assert(dimension < 4, "We only support dimension up to 3");

		output.makeZero();

		callComputeFlux<Flux, Equation, dimension>(equation, conservedVariables, *left, *right, output, *fluxOutput, getNumberOfGhostCells(), waveSpeeds, *reconstruction);
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
