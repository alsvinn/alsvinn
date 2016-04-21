#include <gtest/gtest.h>
#include <iostream>
#include "alsfvm/types.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/numflux/NumericalFluxFactory.hpp"
#include "utils/polyfit.hpp"

using namespace alsfvm;

namespace {
    struct FluxTestParameters {
        const std::string name;
        const std::string platform;

        FluxTestParameters (
            const std::string& name_,
            const std::string& platform_
            )
            :
            name(name_),
    #ifdef ALSVINN_HAVE_CUDA
            platform(platform_)
    #else
            platform("cpu")
    #endif
        {

        }

    };


    std::ostream& operator<<(std::ostream& os, const FluxTestParameters& parameters) {
        os << "\n{\n"
            << "\n\tname = " << parameters.name
            << "\n\tplatform = " << parameters.platform << std::endl << "\n}\n" << std::endl;
        return os;
    }
}
struct BurgersFluxTest : public ::testing::TestWithParam <FluxTestParameters>  {
    const std::string equation = "burgers";

    FluxTestParameters parameters;
    shared_ptr<DeviceConfiguration> deviceConfiguration;
    shared_ptr<memory::MemoryFactory> memoryFactory;



    volume::VolumeFactory volumeFactory;
    shared_ptr<grid::Grid> grid;
    shared_ptr<volume::Volume> conservedVolumeIn;

    shared_ptr<volume::Volume> conservedVolumeOut;

    shared_ptr<simulator::SimulatorParameters> simulatorParameters
        = make_shared<simulator::SimulatorParameters>();

    numflux::NumericalFluxFactory numericalFluxFactory;


    BurgersFluxTest()
        : parameters(GetParam()),
          deviceConfiguration(new DeviceConfiguration(parameters.platform)),
          memoryFactory(new memory::MemoryFactory(deviceConfiguration)),
          volumeFactory(equation, memoryFactory),
          numericalFluxFactory(equation, parameters.name, "none",
                               simulatorParameters, deviceConfiguration)
    {

    }

    void performConvergenceTestConstant() {
        const size_t startK = 10;
        const size_t endK = 20;

        const real constant = 42;
        for (size_t k = startK; k < endK; ++k) {
            const size_t N = 1 << k;
            grid.reset(new grid::Grid({0,0,0}, {1,0,0}, ivec3{N, 1, 1}));
            auto numericalFlux = numericalFluxFactory.createNumericalFlux(*grid);
            conservedVolumeIn = volumeFactory.createConservedVolume(N, 1, 1, 1);
            conservedVolumeOut = volumeFactory.createConservedVolume(N, 1, 1, 1);

            // fill up array
            for(size_t i = 0; i < N+2; ++i) {
                conservedVolumeIn->getScalarMemoryArea("u")->getPointer()[i]
                        = constant;
            }

            rvec3 waveSpeed;
            numericalFlux->computeFlux(*conservedVolumeIn, waveSpeed, false, *conservedVolumeOut);

            for (size_t i = 1; i < N+1; ++i) {
                ASSERT_FLOAT_EQ(0, conservedVolumeOut->getScalarMemoryArea("u")->getPointer()[i]);
            }
        }
    }



    void performConvergenceTestStaircase() {
        const size_t startK = 10;
        const size_t endK = 20;

        const real constant = 42;
        std::vector<double> differences;
        std::vector<double> dx;
        for (size_t k = startK; k < endK; ++k) {
            const size_t N = 1 << k;
            dx.push_back(1.0/N);
            grid.reset(new grid::Grid({0,0,0}, {1,0,0}, ivec3{N, 1, 1}));
            auto numericalFlux = numericalFluxFactory.createNumericalFlux(*grid);
            conservedVolumeIn = volumeFactory.createConservedVolume(N, 1, 1, 1);
            conservedVolumeOut = volumeFactory.createConservedVolume(N, 1, 1, 1);

            // fill up array
            for(size_t i = 0; i < N+2; ++i) {
                conservedVolumeIn->getScalarMemoryArea("u")->getPointer()[i]
                        = N - i;
            }

            rvec3 waveSpeed;
            numericalFlux->computeFlux(*conservedVolumeIn, waveSpeed, false, *conservedVolumeOut);

            double l1ErrorSum = 0.0;
            for (size_t i = 1; i < N+1; ++i) {
               l1ErrorSum += std::abs((N-i+1)*(N-i+1) / 2. - (N-i-1)*(N-i-1) / 2. - conservedVolumeOut->getScalarMemoryArea("u")->getPointer()[i]);
            }

            differences.push_back(l1ErrorSum / N);
        }

        for (auto v : differences) {
            std::cout << v << std::endl;
        }
        auto fit = linearFit(dx, differences);

        ASSERT_FLOAT_EQ(1, fit[0]);


    }

};


TEST_P(BurgersFluxTest, Tests) {
    performConvergenceTestConstant();
    performConvergenceTestStaircase();
}

INSTANTIATE_TEST_CASE_P(Tests,
    BurgersFluxTest,
    ::testing::Values(
    FluxTestParameters("central", "cuda")

    ));
