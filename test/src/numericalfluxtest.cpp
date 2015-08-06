#include <gtest/gtest.h>

#include "alsfvm/types.hpp"
#include "alsfvm/numflux/NumericalFluxFactory.hpp"
#include "alsfvm/numflux/euler/NumericalFluxCPU.hpp"
#include "alsfvm/numflux/euler/HLL.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/equation/euler/AllVariables.hpp"
#include "alsfvm/equation/CellComputerFactory.hpp"

using namespace alsfvm::numflux;
using namespace alsfvm;
using namespace alsfvm::volume;

class NumericalFluxTest : public ::testing::Test {
public:
    std::string equation;
    std::string flux;
    std::string reconstruction;
    std::shared_ptr<DeviceConfiguration> deviceConfiguration;
    NumericalFluxFactory fluxFactory;
    grid::Grid grid;
	std::shared_ptr<memory::MemoryFactory> memoryFactory;
	volume::VolumeFactory volumeFactory;
	const size_t nx;
	const size_t ny;
	const size_t nz;

    NumericalFluxTest()
        : equation("euler"), flux("HLL"), reconstruction("none"),
          deviceConfiguration(new DeviceConfiguration("cpu")),
          fluxFactory(equation, flux, reconstruction, deviceConfiguration),
          grid(rvec3(0,0,0), rvec3(1,1,1), ivec3(20, 20, 20)),
		  memoryFactory(new memory::MemoryFactory(deviceConfiguration)),
		  volumeFactory(equation, memoryFactory), nx(10), ny(10), nz(10)
	{

    }
};

TEST_F(NumericalFluxTest, ConstructionTest) {
    auto numericalFlux = fluxFactory.createNumericalFlux(grid);
}

TEST_F(NumericalFluxTest, ConsistencyTest) {
	// This test that the flux is consistent
	
	auto conservedVariables = volumeFactory.createConservedVolume(nx, ny, nz, 1);
	auto extraVariables = volumeFactory.createExtraVolume(nx, ny, nz, 1);


    for (size_t j = 0; j < nx*ny*nz; j++) {
        conservedVariables->getScalarMemoryArea(0)->getPointer()[j] = 1;
        conservedVariables->getScalarMemoryArea(1)->getPointer()[j] = 1;
        conservedVariables->getScalarMemoryArea(2)->getPointer()[j] = 1;
        conservedVariables->getScalarMemoryArea(3)->getPointer()[j] = 1;
        conservedVariables->getScalarMemoryArea(4)->getPointer()[j] = 10;
    }

    equation::CellComputerFactory cellComputerFactory("cpu", "euler", deviceConfiguration);

    auto computer = cellComputerFactory.createComputer();
    computer->computeExtraVariables(*conservedVariables, *extraVariables);

    ASSERT_TRUE(computer->obeysConstraints(*conservedVariables, *extraVariables));
	auto output = volumeFactory.createConservedVolume(nx, ny, nz, 1);

	for (size_t i = 0; i < output->getNumberOfVariables(); i++) {
		for (size_t j = 0; j < nx*ny*nz; j++) {
			output->getScalarMemoryArea(i)->getPointer()[j] = 1;
		}
	}
	auto numericalFlux = fluxFactory.createNumericalFlux(grid);

	numericalFlux->computeFlux(*conservedVariables, *extraVariables, rvec3(1, 1, 1), *output);

	// Check that output is what we expect
	// Here the flux should be consistent, so we expect that 
	// the difference f(U,Ur)-f(Ul,U) should be zero everywhere.
	for (size_t n = 0; n < output->getNumberOfVariables(); n++) {

		for (size_t k = 1; k < nz - 1; k++) {
			for (size_t j = 1; j < ny - 1; j++) {
				for (size_t i = 1; i < nx - 1; i++) {
					const size_t index = k*nx*ny + j * nx + i;

					ASSERT_EQ(0, output->getScalarMemoryArea(n)->getPointer()[index]);
				}
			}

		}
	}
	
}
