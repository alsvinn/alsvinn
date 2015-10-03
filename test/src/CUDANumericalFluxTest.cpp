#include <gtest/gtest.h>

#include "alsfvm/types.hpp"
#include "alsfvm/numflux/NumericalFluxFactory.hpp"
#include "alsfvm/numflux/euler/NumericalFluxCPU.hpp"
#include "alsfvm/numflux/euler/HLL.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/equation/euler/AllVariables.hpp"
#include "alsfvm/equation/CellComputerFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/boundary/BoundaryFactory.hpp"
#include "alsfvm/cuda/cuda_utils.hpp" 

using namespace alsfvm::numflux;
using namespace alsfvm;
using namespace alsfvm::volume;

class CUDANumericalFluxTest : public ::testing::Test {
public:
	std::string equation;
	std::string flux;
	std::string reconstruction;
	alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration;
	alsfvm::shared_ptr<DeviceConfiguration> deviceConfigurationCPU;
	NumericalFluxFactory fluxFactory;
	NumericalFluxFactory fluxFactoryCPU;
	grid::Grid grid;
	alsfvm::shared_ptr<memory::MemoryFactory> memoryFactory;
	alsfvm::shared_ptr<memory::MemoryFactory> memoryFactoryCPU;
	volume::VolumeFactory volumeFactory;
	volume::VolumeFactory volumeFactoryCPU;
	const size_t nx;
	const size_t ny;
	const size_t nz;

	CUDANumericalFluxTest()
		: equation("euler"), flux("HLL"), reconstruction("none"),
		deviceConfiguration(new DeviceConfiguration("cuda")),
		deviceConfigurationCPU(new DeviceConfiguration("cpu")),
		fluxFactory(equation, flux, reconstruction, deviceConfiguration),
		fluxFactoryCPU(equation, flux, reconstruction, deviceConfigurationCPU),
		grid(rvec3(0, 0, 0), rvec3(1, 1, 1), ivec3(10, 10, 1)),
		memoryFactory(new memory::MemoryFactory(deviceConfiguration)),
		memoryFactoryCPU(new memory::MemoryFactory(deviceConfigurationCPU)),
		volumeFactory(equation, memoryFactory), volumeFactoryCPU(equation, memoryFactoryCPU), 
		nx(10), ny(10), nz(1)
	{

	}
};

TEST_F(CUDANumericalFluxTest, ConstructionTest) {
	auto numericalFlux = fluxFactory.createNumericalFlux(grid);
}

TEST_F(CUDANumericalFluxTest, ConsistencyTest) {
	// This test that the flux is consistent

	auto conservedVariables = volumeFactory.createConservedVolume(nx, ny, nz, 1);
	auto extraVariables = volumeFactory.createExtraVolume(nx, ny, nz, 1);

	
	boundary::BoundaryFactory boundaryFactory("neumann", deviceConfiguration);

	auto boundary = boundaryFactory.createBoundary(1);
	auto conservedVariablesCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, 1);
	auto extraVariablesCPU = volumeFactoryCPU.createExtraVolume(nx, ny, nz, 1);
	volume::fill_volume<equation::euler::ConservedVariables>(*conservedVariablesCPU, grid,
		[](real x, real y, real z, equation::euler::ConservedVariables& out) {
		out.rho = 1;
		out.m.x = 1;
		out.m.y = 1;
		out.m.z = 1;
		out.E = 10;

	});
	
	conservedVariablesCPU->copyTo(*conservedVariables);
	boundary->applyBoundaryConditions(*conservedVariables, grid);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	equation::CellComputerFactory cellComputerFactory("cuda", "euler", deviceConfiguration);

	auto computer = cellComputerFactory.createComputer();
	computer->computeExtraVariables(*conservedVariables, *extraVariables);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	ASSERT_TRUE(computer->obeysConstraints(*conservedVariables, *extraVariables));
	auto output = volumeFactory.createConservedVolume(nx, ny, nz, 1);
	
	
	for (size_t i = 0; i < output->getNumberOfVariables(); i++) {
		std::vector<real> hostOutput(output->getScalarMemoryArea(i)->getSize(), 44);
		output->getScalarMemoryArea(i)->copyFromHost(hostOutput.data(), hostOutput.size());
	}
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	auto numericalFlux = fluxFactory.createNumericalFlux(grid);
	ASSERT_EQ(1, numericalFlux->getNumberOfGhostCells());
	auto outputCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, 1);
	output->copyTo(*outputCPU);
	for (size_t n = 0; n < output->getNumberOfVariables(); n++) {

		for (size_t k = 0; k < nz; k++) {
			for (size_t j = 1; j < ny - 1; j++) {
				for (size_t i = 1; i < nx - 1; i++) {
					ASSERT_EQ(44, outputCPU->getScalarMemoryArea(n)->getView().at(i, j, k))
						<< "Initial data failed at. (" << i << ", " << j << ", " << k << ")";
				}
			}

		}
	}

	numericalFlux->computeFlux(*conservedVariables, rvec3(1, 1, 1), *output);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	CUDA_SAFE_CALL(cudaGetLastError());
	// Check that output is what we expect
	// Here the flux should be consistent, so we expect that 
	// the difference f(U,Ur)-f(Ul,U) should be zero everywhere.
	output->copyTo(*outputCPU);

	for (size_t n = 0; n < output->getNumberOfVariables(); n++) {

		for (size_t k = 0; k < nz; k++) {
			for (size_t j = 1; j < ny - 1; j++) {
				for (size_t i = 1; i < nx - 1; i++) {

					
					ASSERT_NEAR(0, outputCPU->getScalarMemoryArea(n)->getView().at(i, j, k), 1e-30)
						<< "Consistency check failed at (" << i << ", " << j << ", " << k << ")";
				}
			}

		}
	}

}

TEST_F(CUDANumericalFluxTest, CompareAgainstCPU) {
	// This test that the flux is consistent

	auto conservedVariables = volumeFactory.createConservedVolume(nx, ny, nz, 1);
	auto extraVariables = volumeFactory.createExtraVolume(nx, ny, nz, 1);


	boundary::BoundaryFactory boundaryFactory("neumann", deviceConfiguration);

	auto boundary = boundaryFactory.createBoundary(1);

	boundary::BoundaryFactory boundaryFactoryCPU("neumann", deviceConfigurationCPU);

	auto boundaryCPU = boundaryFactoryCPU.createBoundary(1);
	auto conservedVariablesCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, 1);
	auto extraVariablesCPU = volumeFactoryCPU.createExtraVolume(nx, ny, nz, 1);
	// We set every other var to 1,1,1,10 and 2,2,2,2,40.
	int switchCounter = 0;
	volume::fill_volume<equation::euler::ConservedVariables>(*conservedVariablesCPU, grid,
		[&switchCounter](real x, real y, real z, equation::euler::ConservedVariables& out) {
		
		if (switchCounter++ % 2) {
			out.rho = 1;
			out.m.x = 1;
			out.m.y = 1;
			out.m.z = 1;
			out.E = 10;
		}
		else {
			out.rho = 2;
			out.m.x = 2;
			out.m.y = 2;
			out.m.z = 2;
			out.E = 40;
		}
	});
	boundaryCPU->applyBoundaryConditions(*conservedVariablesCPU, grid);

	conservedVariablesCPU->copyTo(*conservedVariables);
	boundary->applyBoundaryConditions(*conservedVariables, grid);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	equation::CellComputerFactory cellComputerFactory("cuda", "euler", deviceConfiguration);

	
	auto output = volumeFactory.createConservedVolume(nx, ny, nz, 1);
	auto outputCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, 1);
	// Set default values
	for (size_t i = 0; i < output->getNumberOfVariables(); i++) {
		std::vector<real> hostOutput(output->getScalarMemoryArea(i)->getSize(), 44);
		output->getScalarMemoryArea(i)->copyFromHost(hostOutput.data(), hostOutput.size());
		outputCPU->getScalarMemoryArea(i)->copyFromHost(hostOutput.data(), hostOutput.size());
	}
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	auto numericalFlux = fluxFactory.createNumericalFlux(grid);
	auto numericalFluxCPU = fluxFactoryCPU.createNumericalFlux(grid);
	ASSERT_EQ(1, numericalFlux->getNumberOfGhostCells());
	ASSERT_EQ(1, numericalFluxCPU->getNumberOfGhostCells());
	auto outputCPUfromGPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, 1);
	output->copyTo(*outputCPUfromGPU);


	for (size_t n = 0; n < output->getNumberOfVariables(); n++) {

		// Check that the output has been properly reset before we run
		for (size_t k = 0; k < nz; k++) {
			for (size_t j = 1; j < ny - 1; j++) {
				for (size_t i = 1; i < nx - 1; i++) {
					
					ASSERT_EQ(44, outputCPUfromGPU->getScalarMemoryArea(n)->getView().at(i, j, k))
						<< "Initial data failed at. (" << i << ", " << j << ", " << k << ")";
					ASSERT_EQ(44, outputCPU->getScalarMemoryArea(n)->getView().at(i, j, k))
						<< "Initial data failed at. (" << i << ", " << j << ", " << k << ")";
				}
			}

		}

		std::vector<real> fromGPU(conservedVariablesCPU->getScalarMemoryArea(0)->getSize());

		// Check that we start with the same data
		conservedVariables->getScalarMemoryArea(n)->copyToHost(fromGPU.data(), fromGPU.size());
		for (size_t k = 0; k < nz; k++) {
			for (size_t j = 0; j < ny; j++) {
				for (size_t i = 0; i < nx; i++) {
					auto view = conservedVariablesCPU->getScalarMemoryArea(n)->getView();
					ASSERT_EQ(fromGPU[view.index(i, j, k)], view.at(i, j, k));
				}
			}

		}
	}

	numericalFlux->computeFlux(*conservedVariables, rvec3(1, 1, 1), *output);
	numericalFluxCPU->computeFlux(*conservedVariablesCPU, rvec3(1, 1, 1), *outputCPU);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	CUDA_SAFE_CALL(cudaGetLastError());
	// Check that output is what we expect
	// Here the flux should be consistent, so we expect that 
	// the difference f(U,Ur)-f(Ul,U) should be zero everywhere.
	output->copyTo(*outputCPUfromGPU);

	for (size_t n = 0; n < output->getNumberOfVariables(); n++) {

		for (size_t k = 0; k < nz; k++) {
			for (size_t j = 2; j < ny - 1; j++) {
				for (size_t i = 2; i < nx - 1; i++) {

					if (std::abs(outputCPU->getScalarMemoryArea(n)->getView().at(i, j, k) -
						outputCPUfromGPU->getScalarMemoryArea(n)->getView().at(i, j, k)) > 1e-8) {
						for (size_t var = 0; var < output->getNumberOfVariables(); ++var) {
							std::cout << "CPU(" <<outputCPU->getName(var) << ") = " << outputCPU->getScalarMemoryArea(n)->getView().at(i, j, k) << std::endl;
							std::cout << "GPU(" << outputCPU->getName(var) << ") = " << outputCPUfromGPU->getScalarMemoryArea(n)->getView().at(i, j, k) << std::endl;
						}
						std::cout << "Consistency check failed at (" << i << ", " << j << ", " << k << ")" << std::endl;

					}
					
					ASSERT_NEAR(outputCPU->getScalarMemoryArea(n)->getView().at(i, j, k),
						outputCPUfromGPU->getScalarMemoryArea(n)->getView().at(i, j, k), 1e-8)
						<< "Consistency check failed at (" << i << ", " << j << ", " << k << ")";
				}
			}

		}
	}

}
