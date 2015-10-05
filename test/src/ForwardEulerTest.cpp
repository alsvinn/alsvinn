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
		real dt;
		ODENumericalFlux(real dt) : dt(dt) {}
		size_t getNumberOfGhostCells() { return 0; }

        void computeFlux(const volume::Volume& conservedVariables,
			rvec3& waveSpeeds, bool computeWaveSpeeds, 
			volume::Volume& output) 
		{
            output.getScalarMemoryArea(0)->getPointer()[0] = dt * conservedVariables.getScalarMemoryArea(0)->getPointer()[0];
			waveSpeeds = rvec3(1, 0, 0);
		}
	};
}
TEST(ForwardEulerTest, ConvergenceTest) {

	// We test that if we integrate the system
	// du/dt = u
	// u(0)=  1
	// we will get an approximation to exp(1) at u(1)


	std::vector<std::string> variableNames = { "u" };

	const size_t nx = 1;
	const size_t ny = 1;
	const size_t nz = 1;

	auto configuration = alsfvm::make_shared<alsfvm::DeviceConfiguration>("cpu");

	auto factory = alsfvm::make_shared<alsfvm::memory::MemoryFactory>(configuration);

    alsfvm::shared_ptr<alsfvm::volume::Volume> volumeIn(new alsfvm::volume::Volume(variableNames, factory, nx, ny, nz));
    alsfvm::shared_ptr<alsfvm::volume::Volume> volumeOut(new alsfvm::volume::Volume(variableNames, factory, nx, ny, nz));


	// Start with u(0)=1
    volumeIn->getScalarMemoryArea(0)->getPointer()[0] = 1;

	const size_t N = 100000;
	const real dt = real(1) / real(N);
	alsfvm::shared_ptr<NumericalFlux> flux(new ODENumericalFlux(dt));
	ForwardEuler integrator(flux);
	for (size_t i = 0; i < N; i++) {
		// First timestep we use input as input and output as output, 
		// but then on the second timestep we need to reverse the roles,
		// and then switch every other timstep
		if (i % 2) {
			// Note that we do not care about spatial resolution here
            integrator.performSubstep({volumeOut}, rvec3(1, 1, 1), dt, 1, *volumeIn, 0, simulator::TimestepInformation());
		}
		else {
			// Note that we do not care about spatial resolution here
            integrator.performSubstep({volumeIn}, rvec3(1, 1, 1), dt, 1, *volumeOut, 0, simulator::TimestepInformation());
		}

	}

    ASSERT_NEAR(volumeOut->getScalarMemoryArea(0)->getPointer()[0], std::exp(1), 1e-4);
}
