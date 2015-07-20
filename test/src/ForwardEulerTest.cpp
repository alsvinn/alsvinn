#include <gtest/gtest.h>
#include "alsfvm/integrator/ForwardEuler.hpp"
#include <cmath>
using namespace alsfvm;
using namespace alsfvm::integrator;
using namespace alsfvm::numflux;

namespace {

	// Represents the system du/dt = u;
	class ODENumericalFlux : public NumericalFlux {
	public:
		size_t getNumberOfGhostCells() { return 0; }

		void computeFlux(const volume::Volume& conservedVariables,
			const volume::Volume& extraVariables,
			const rvec3& cellLengths,
			volume::Volume& output) 
		{
			output.getScalarMemoryArea(0)->getPointer()[0] = conservedVariables.getScalarMemoryArea(0)->getPointer()[0];
		}
	};
}
TEST(ForwardEulerTest, ConvergenceTest) {

	// We test that if we integrate the system
	// du/dt = u
	// u(0)=  1
	// we will get an approximation to exp(1) at u(1)
	std::shared_ptr<NumericalFlux> flux(new ODENumericalFlux);

	std::vector<std::string> variableNames = { "u" };

	const size_t nx = 1;
	const size_t ny = 1;
	const size_t nz = 1;

	auto configuration = std::make_shared<alsfvm::DeviceConfiguration>();

	auto factory = std::make_shared<alsfvm::memory::MemoryFactory>("HostMemory",
		configuration);

	alsfvm::volume::Volume volumeIn(variableNames, factory, nx, ny, nz);
	alsfvm::volume::Volume volumeOut(variableNames, factory, nx, ny, nz);


	// Start with u(0)=1
	volumeIn.getScalarMemoryArea(0)->getPointer()[0] = 1;

	const size_t N = 100000;
	const real dt = real(1) / real(N);
	ForwardEuler integrator(flux);
	for (size_t i = 0; i < N; i++) {
		// First timestep we use input as input and output as output, 
		// but then on the second timestep we need to reverse the roles,
		// and then switch every other timstep
		if (i % 2) {
			// Note that we do not care about spatial resolution here
			integrator.performSubstep(volumeOut, volumeOut, rvec3(0, 0, 0), dt, volumeIn);
		}
		else {
			// Note that we do not care about spatial resolution here
			integrator.performSubstep(volumeIn, volumeIn, rvec3(0, 0, 0), dt, volumeOut);
		}

	}

	ASSERT_LT(std::abs(volumeOut.getScalarMemoryArea(0)->getPointer()[0] - std::exp(1)), 1e-4);
}
