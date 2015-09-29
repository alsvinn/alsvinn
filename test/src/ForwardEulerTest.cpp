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
			const rvec3& cellLengths,
			volume::Volume& output) 
		{
            output.getScalarMemoryArea(0)->getPointer()[0] = cellLengths.x * conservedVariables.getScalarMemoryArea(0)->getPointer()[0];
		}
	};
}
TEST(ForwardEulerTest, ConvergenceTest) {

	// We test that if we integrate the system
	// du/dt = u
	// u(0)=  1
	// we will get an approximation to exp(1) at u(1)
	boost::shared_ptr<NumericalFlux> flux(new ODENumericalFlux);

	std::vector<std::string> variableNames = { "u" };

	const size_t nx = 1;
	const size_t ny = 1;
	const size_t nz = 1;

	auto configuration = boost::make_shared<alsfvm::DeviceConfiguration>("cpu");

	auto factory = boost::make_shared<alsfvm::memory::MemoryFactory>(configuration);

    boost::shared_ptr<alsfvm::volume::Volume> volumeIn(new alsfvm::volume::Volume(variableNames, factory, nx, ny, nz));
    boost::shared_ptr<alsfvm::volume::Volume> volumeOut(new alsfvm::volume::Volume(variableNames, factory, nx, ny, nz));


	// Start with u(0)=1
    volumeIn->getScalarMemoryArea(0)->getPointer()[0] = 1;

	const size_t N = 100000;
	const real dt = real(1) / real(N);
	ForwardEuler integrator(flux);
	for (size_t i = 0; i < N; i++) {
		// First timestep we use input as input and output as output, 
		// but then on the second timestep we need to reverse the roles,
		// and then switch every other timstep
		if (i % 2) {
			// Note that we do not care about spatial resolution here
            integrator.performSubstep({volumeOut}, rvec3(1, 1, 1), dt, *volumeIn, 0);
		}
		else {
			// Note that we do not care about spatial resolution here
            integrator.performSubstep({volumeIn}, rvec3(1, 1, 1), dt, *volumeOut, 0);
		}

	}

    ASSERT_LT(std::abs(volumeOut->getScalarMemoryArea(0)->getPointer()[0] - std::exp(1)), 1e-4);
}
