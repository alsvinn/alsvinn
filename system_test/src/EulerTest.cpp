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
#include <array>

using namespace alsfvm;
using namespace alsfvm::grid;
using namespace alsfvm::memory;
using namespace alsfvm::volume;
using namespace alsfvm::numflux;
using namespace alsfvm::equation::euler;
using namespace alsfvm::equation;

TEST(EulerTest, ShockTubeTest) {
	const size_t N = 128;
	const real T = 0.06;

	Grid grid(rvec3(0, 0, 0), rvec3(1, 1, 1), ivec3(N, N, 1));

	auto deviceConfiguration = std::make_shared<DeviceConfiguration>("cpu");

	auto memoryFactory = std::make_shared<MemoryFactory>(deviceConfiguration);

	VolumeFactory volumeFactory("euler", memoryFactory);

	NumericalFluxFactory fluxFactory("euler", "HLL", "none", deviceConfiguration);
	CellComputerFactory cellComputerFactory("cpu", "euler", deviceConfiguration);
	auto conserved1 = volumeFactory.createConservedVolume(N, N, 1);
	auto conserved2 = volumeFactory.createConservedVolume(N, N, 1);

	auto extra1 = volumeFactory.createExtraVolume(N, N, 1);
	auto extra2 = volumeFactory.createExtraVolume(N, N, 1);

	fill_volume<ConservedVariables, ExtraVariables>(*conserved1, *extra1, grid, 
		[](real x, real y, real z, ConservedVariables& u, ExtraVariables& v) {

		if (x < 0.4) {
			u.rho = 3.86859;
			v.u.x = 11.2536;
			v.p = 167.345;
		} 
		else {
			real r = pow(x - 0.25, 2) + pow(y - 0.5, 2);
			//real phi = (x - 0.25) / sqrt(r);
			real r_max = 4*pow(0.13, 2);
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
	real dt = 0.4 * grid.getCellLengths().x / cellComputer->computeMaxWaveSpeed(*conserved1, *extra1);
	io::HDF5Writer writer("EulerTest");
	writer.write(*conserved1, *extra1, grid, simulator::TimestepInformation());

	int i = 0;
	while (t < T) {
		fowardEuler.performSubstep(*conserved1, *extra1, grid.getCellLengths(), dt, *conserved2);
		conserved1.swap(conserved2);
		std::array<real*, 4> conservedPointers = {
			conserved1->getScalarMemoryArea(0)->getPointer(),
			conserved1->getScalarMemoryArea(1)->getPointer(),
			conserved1->getScalarMemoryArea(2)->getPointer(),
			conserved1->getScalarMemoryArea(3)->getPointer()
		};

		for (size_t x = 0; x < N; x++) {
			const size_t index1 = x;
			const size_t index2 = N + x;
			const size_t index3 = (N - 1)*N + x;
			const size_t index4 = (N - 2)*N + x;
			for (size_t i = 0; i < conservedPointers.size(); i++) {
				conservedPointers[i][index1] = conservedPointers[i][index2];
				conservedPointers[i][index3] = conservedPointers[i][index4];
			}
		}

		for (size_t y = 0; y < N; y++) {
			const size_t index1 = y*N;
			const size_t index2 = y*N + 1;
			const size_t index3 = y*N + (N - 1);
			const size_t index4 = y*N + (N - 2);
			for (size_t i = 0; i < conservedPointers.size(); i++) {
				conservedPointers[i][index1] = conservedPointers[i][index2];
				conservedPointers[i][index3] = conservedPointers[i][index4];
			}
		}
		
		cellComputer->computeExtraVariables(*conserved1, *extra1);
		t += dt;
		dt = 0.4 * grid.getCellLengths().x / cellComputer->computeMaxWaveSpeed(*conserved1, *extra1);

		i++;

		if (i % 20) {
			writer.write(*conserved1, *extra1, grid, simulator::TimestepInformation());
		}


	}

	
	
}