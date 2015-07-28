#include <gtest/gtest.h>
#include "alsfvm/grid/Grid.hpp"
#include "alsfvm/memory/MemoryFactory.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/numflux/NumericalFluxFactory.hpp"
#include "alsfvm/equation/CellComputerFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/integrator/ForwardEuler.hpp"
#include "alsfvm/io/HDF5Writer.hpp"
#include "alsfvm/boundary/BoundaryFactory.hpp"
#include <array>

using namespace alsfvm;
using namespace alsfvm::grid;
using namespace alsfvm::memory;
using namespace alsfvm::volume;
using namespace alsfvm::numflux;
using namespace alsfvm::equation::euler;
using namespace alsfvm::equation;
using namespace alsfvm::boundary;

TEST(EulerTest, ShockTubeTest) {
    const size_t N = 256;
	const real T = 0.06;
    const real cfl = 0.6;

	Grid grid(rvec3(0, 0, 0), rvec3(1, 1, 1), ivec3(N, N, 1));

	auto deviceConfiguration = std::make_shared<DeviceConfiguration>("cpu");

	auto memoryFactory = std::make_shared<MemoryFactory>(deviceConfiguration);

	VolumeFactory volumeFactory("euler", memoryFactory);

	NumericalFluxFactory fluxFactory("euler", "HLL", "none", deviceConfiguration);
	CellComputerFactory cellComputerFactory("cpu", "euler", deviceConfiguration);
	BoundaryFactory boundaryFactory("neumann", deviceConfiguration);
	auto conserved1 = volumeFactory.createConservedVolume(N, N, 1);
	auto conserved2 = volumeFactory.createConservedVolume(N, N, 1);

	auto extra1 = volumeFactory.createExtraVolume(N, N, 1);
	auto extra2 = volumeFactory.createExtraVolume(N, N, 1);

	fill_volume<ConservedVariables, ExtraVariables>(*conserved1, *extra1, grid, 
		[](real x, real y, real z, ConservedVariables& u, ExtraVariables& v) {

        if (x < 0.04) {
			u.rho = 3.86859;
			v.u.x = 11.2536;
			v.p = 167.345;
		} 
		else {
			real r = pow(x - 0.25, 2) + pow(y - 0.5, 2);
            real r_max = pow(0.15, 2);

            if (r <= r_max) {
				u.rho = 10.0;
			}
			else {
				u.rho = 1.0;
			}
			v.p = 1.0;
		}
		u.m = u.rho * v.u;
		u.E = v.p / (GAMMA - 1) + 0.5*u.rho*v.u.dot(v.u);
	});
	auto numericalFlux = fluxFactory.createNumericalFlux(grid);
	integrator::ForwardEuler fowardEuler(numericalFlux);

	auto cellComputer = cellComputerFactory.createComputer();

	real t = 0;
    real dt = cfl * grid.getCellLengths().x / cellComputer->computeMaxWaveSpeed(*conserved1, *extra1);
	io::HDF5Writer writer("EulerTest");
	writer.write(*conserved1, *extra1, grid, simulator::TimestepInformation());
    cellComputer->computeExtraVariables(*conserved1, *extra1);
    ASSERT_TRUE(cellComputer->obeysConstraints(*conserved1, *extra1));
	int i = 0;
    size_t numberOfTimesteps = 0;
	auto boundary = boundaryFactory.createBoundary(numericalFlux->getNumberOfGhostCells());
	while (t < T) {
        numberOfTimesteps++;
		fowardEuler.performSubstep(*conserved1, *extra1, grid.getCellLengths(), dt, *conserved2);
		conserved1.swap(conserved2);
        std::array<real*, 5> conservedPointers = {
			conserved1->getScalarMemoryArea(0)->getPointer(),
			conserved1->getScalarMemoryArea(1)->getPointer(),
			conserved1->getScalarMemoryArea(2)->getPointer(),
            conserved1->getScalarMemoryArea(3)->getPointer(),
            conserved1->getScalarMemoryArea(4)->getPointer()
		};

        std::array<real*, 5> extraPointers = {
            extra1->getScalarMemoryArea(0)->getPointer(),
            extra1->getScalarMemoryArea(1)->getPointer(),
            extra1->getScalarMemoryArea(2)->getPointer(),
            extra1->getScalarMemoryArea(3)->getPointer()

        };

		boundary->applyBoundaryConditions(*conserved1, grid);

		
		// Intense error checking below. We basically check that the output is sane,
		// this doubles the work of cellComputer->obeysConstraints.
        for(size_t i = 0; i < conserved1->getScalarMemoryArea(0)->getSize();i++) {
            // Check that density and pressure is positive
            ASSERT_GT(conservedPointers[0][i], 0);
            ASSERT_GT(extraPointers[0][i], 0);

            for(size_t j = 0; j < 5; j++) {
                if (std::isnan(conservedPointers[j][i])) {
                    std::cout << "(conserved) nan at i = " << i;
                    std::cout << " variable " << conserved1->getName(j) << std::endl;
                }
                ASSERT_FALSE(std::isnan(conservedPointers[j][i]));

                if (std::isinf(conservedPointers[j][i])) {
                    std::cout << "(conserved) inf at i = " << i;
                    std::cout << " variable " << conserved1->getName(j) << std::endl;
                }
                ASSERT_FALSE(std::isinf(conservedPointers[j][i]));
            }

            for(size_t j = 0; j < 4; j++) {
                if (std::isnan(extraPointers[j][i])) {
                    std::cout << "(extra ) nan at i = " << i;
                    std::cout << " variable " << extra1->getName(j) << std::endl;
                }
                ASSERT_FALSE(std::isnan(extraPointers[j][i]));

                if (std::isinf(extraPointers[j][i])) {
                    std::cout << "*( extra ) inf at i = " << i;
                    std::cout << " variable " << extra1->getName(j) << std::endl;
                }
                ASSERT_FALSE(std::isinf(extraPointers[j][i]));
            }
        }

        cellComputer->computeExtraVariables(*conserved1, *extra1);

        ASSERT_TRUE(cellComputer->obeysConstraints(*conserved1, *extra1));
		t += dt;
        dt = cfl * grid.getCellLengths().x / cellComputer->computeMaxWaveSpeed(*conserved1, *extra1);
        ASSERT_FALSE(std::isnan(dt));

        ASSERT_GT(dt, 0);
		i++;

		if (i % 20) {
            writer.write(*conserved1, *extra1, grid, simulator::TimestepInformation());
		}


	}

    std::cout << "Number of timesteps used: " << numberOfTimesteps << std::endl;
    writer.write(*conserved1, *extra1, grid, simulator::TimestepInformation());
	
	
}
