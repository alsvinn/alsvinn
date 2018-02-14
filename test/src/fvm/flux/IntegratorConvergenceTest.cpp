#include <gtest/gtest.h>
#include "alsfvm/integrator/IntegratorFactory.hpp"
#include <cmath>
#include "utils/polyfit.hpp"
using namespace alsfvm;
using namespace alsfvm::integrator;
using namespace alsfvm::numflux;

namespace {

// Represents the system du/dt = u;
class ODESystem : public System {
    public:
        real dt;
        ODESystem(real dt) : dt(dt) {}
        size_t getNumberOfGhostCells() {
            return 0;
        }

        void operator()( volume::Volume& conservedVariables,
            rvec3& waveSpeeds, bool computeWaveSpeeds,
            volume::Volume& output) {
            output.getScalarMemoryArea(0)->getPointer()[0] = dt *
                conservedVariables.getScalarMemoryArea(0)->getPointer()[0];
            waveSpeeds = rvec3(1, 0, 0);
        }
};
}
struct IntegratorParameters {
    IntegratorParameters(const std::string& name,
        double expectedConvergenceRate) :
        name(name),
        expectedConvergenceRate(expectedConvergenceRate)
    {}

    std::string name;
    double expectedConvergenceRate;
};

std::ostream& operator<<(std::ostream& os,
    const IntegratorParameters& parameters) {
    os << "\n{\n\texpectedConvergenceRate = " << parameters.expectedConvergenceRate

        << "\n\tname = " << parameters.name
        << std::endl << "}" << std::endl;
    return os;
}
class IntegratorConvergenceTest : public ::testing::TestWithParam
    <IntegratorParameters> {
    public:

        IntegratorConvergenceTest() :
            name(GetParam().name),
            expectedConvergenceRate(GetParam().expectedConvergenceRate) {
        }

        std::string name;
        double expectedConvergenceRate;
};

TEST_P(IntegratorConvergenceTest, ConvergenceTest) {

    // We test that if we integrate the system
    // du/dt = u
    // u(0)=  1
    // we will get an approximation to exp(1) at u(1)


    std::vector<std::string> variableNames = { "u" };

    const size_t nx = 1;
    const size_t ny = 1;
    const size_t nz = 1;

    auto configuration = alsfvm::make_shared<alsfvm::DeviceConfiguration>("cpu");

    auto factory = alsfvm::make_shared<alsfvm::memory::MemoryFactory>
        (configuration);


    std::vector<real> errors;
    std::vector<real> resolutions;

    for (size_t k = 3; k < 9; ++k) {
        const size_t N = (2 << k);
        const real dt = real(1) / real(N);
        alsfvm::shared_ptr<System> flux(new ODESystem(dt));

        IntegratorFactory integratorFactory(name);
        auto integrator = integratorFactory.createIntegrator(flux);
        std::vector<alsfvm::shared_ptr<alsfvm::volume::Volume> >
        volumes(integrator->getNumberOfSubsteps() + 1);

        for (auto& volume : volumes) {
            volume.reset(new alsfvm::volume::Volume(variableNames, factory, nx, ny, nz));
            volume->getScalarMemoryArea(0)->getPointer()[0] = 1;
        }

        double t = 0;
        simulator::TimestepInformation timestepInformation;
        const double cfl = 1; // To keep timestep constant

        for (size_t i = 0; i < N; i++) {
            for (size_t substep = 0; substep < integrator->getNumberOfSubsteps();
                ++substep) {
                auto& currentVolume = volumes[substep + 1];

                // Note that we do not care about spatial resolution here
                integrator->performSubstep(volumes, rvec3(1, 1, 1), 1, cfl, *currentVolume,
                    substep, timestepInformation);

            }

            timestepInformation.incrementTime(dt);
            volumes.back().swap(volumes.front());
            t += dt;
        }

        ASSERT_EQ(t, 1);
        const double error = std::abs(std::exp(1) -
                volumes.front()->getScalarMemoryArea(0)->getPointer()[0]);

        std::cout << "[" << N << ", " << error << "]," << std::endl;
        errors.push_back(std::log(error));
        resolutions.push_back(std::log(N));


    }



    ASSERT_LE(expectedConvergenceRate, -linearFit(resolutions, errors)[0]);

}

INSTANTIATE_TEST_CASE_P(IntegratorConvergenceTests,
    IntegratorConvergenceTest,
    ::testing::Values(
        IntegratorParameters("forwardeuler", 0.9),
        IntegratorParameters("rungekutta2", 1.9),
        IntegratorParameters("rungekutta3", 2.9),
        IntegratorParameters("rungekutta4", 3.9)

    ));
