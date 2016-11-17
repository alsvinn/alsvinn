#include <gtest/gtest.h>
#include "alsfvm/types.hpp"
#include "alsfvm/diffusion/DiffusionFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "utils/polyfit.hpp"
#include "alsfvm/boundary/BoundaryFactory.hpp"
#include <iostream>

using namespace alsfvm;

struct DiffusionParameters {
    std::string platform;
    std::string equation;
    std::string diffusion;
    std::string reconstruction;

    double expectedConvergenceRate;


    alsfvm::shared_ptr<memory::MemoryFactory> memoryFactory;


    DiffusionParameters(const std::string& platform, const std::string& equation, const std::string& diffusion,
        const std::string& reconstruction,
        double expectedConvergenceRate) 
        : platform(platform), equation(equation), diffusion(diffusion), reconstruction(reconstruction), expectedConvergenceRate(expectedConvergenceRate)
    {
        
    }

   
};

std::ostream& operator<<(std::ostream& os, const DiffusionParameters& parameters) {
    os << "\n{\n\texpectedConvergenceRate = " << parameters.expectedConvergenceRate

        << "\n\tdiffusion = " << parameters.diffusion
        << "\n\treconstruction = " << parameters.reconstruction
        << "\n\tplatform = " << parameters.platform << std::endl << "}" << std::endl;
    return os;
}
class TecnoDiffusionTest : public ::testing::TestWithParam <DiffusionParameters> {
public:

    TecnoDiffusionTest()
     :   parameters(GetParam())
    {
        deviceConfiguration.reset(new DeviceConfiguration(parameters.platform));
        deviceConfigurationCPU.reset(new DeviceConfiguration("cpu"));

        memoryFactory.reset(new memory::MemoryFactory(deviceConfiguration));
        memoryFactoryCPU.reset(new memory::MemoryFactory(deviceConfigurationCPU));

        volumeFactory.reset(new volume::VolumeFactory(parameters.equation, memoryFactory));
        volumeFactoryCPU.reset(new volume::VolumeFactory(parameters.equation, memoryFactoryCPU));

        boundaryFactory.reset(new boundary::BoundaryFactory("periodic", deviceConfiguration));
    }

    DiffusionParameters parameters;

    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration;
    alsfvm::shared_ptr<DeviceConfiguration> deviceConfigurationCPU;

    alsfvm::shared_ptr<memory::MemoryFactory> memoryFactory;
    alsfvm::shared_ptr<memory::MemoryFactory> memoryFactoryCPU;

    alsfvm::shared_ptr<volume::VolumeFactory> volumeFactory;
    alsfvm::shared_ptr<volume::VolumeFactory> volumeFactoryCPU;

    alsfvm::simulator::SimulatorParameters simulatorParameters;

    alsfvm::shared_ptr<boundary::BoundaryFactory> boundaryFactory;
  
    std::tuple<grid::Grid, alsfvm::shared_ptr<volume::Volume>, alsfvm::shared_ptr<diffusion::DiffusionOperator> > makeSetup(size_t nx) {
        auto grid = grid::Grid(rvec3(0., 0., 0.), rvec3(1., 0., 0.), ivec3(nx, 1, 1));
        
        diffusion::DiffusionFactory diffusionFactory;

        auto diffusionOperator = diffusionFactory.createDiffusionOperator(parameters.equation, parameters.diffusion,
            parameters.reconstruction, grid, simulatorParameters,deviceConfiguration, memoryFactory,
            *volumeFactory);

        auto boundary = boundaryFactory->createBoundary(diffusionOperator->getNumberOfGhostCells());

        auto volumeCPU = volumeFactoryCPU->createConservedVolume(nx, 1, 1, diffusionOperator->getNumberOfGhostCells());
        volumeCPU->makeZero();
        
        auto f = [](real x) {
            return 0.5*sin(2 * M_PI * x) + 1;
        };

        // Integral of f / dx
        // where dx = b - a
        auto averageIntegralF = [](real a, real b) {
            return 0.5*(-cos(2 * M_PI * b) + cos(2 * M_PI * a)) / (2 * M_PI * (b - a)) + 1;
        };

        double dx = grid.getCellLengths().x;
        volume::for_each_midpoint(*volumeCPU, grid, [&](real x, real y, real z, size_t index) {
            real a = index * dx;
            real b = (index + 1) * dx;
            for (size_t i = 0; i < volumeCPU->getNumberOfVariables(); ++i) {
                volumeCPU->getScalarMemoryArea(i)->getPointer()[index] = averageIntegralF(a,b);
            }
        });

        auto volume = volumeFactory->createConservedVolume(nx, 1, 1, diffusionOperator->getNumberOfGhostCells());

        volumeCPU->copyTo(*volume);

        boundary->applyBoundaryConditions(*volume, grid);
        return std::make_tuple(grid, volume, diffusionOperator);
    }
};


TEST_P(TecnoDiffusionTest, OrderTest) {

    std::vector<real> errors;
    std::vector<real> resolutions;

    const int minK = 4;
    const int maxK = 12;
    for (int k = minK; k < maxK; ++k) {
        const int nx = 1 << k;

        auto data = makeSetup(nx);
        auto grid = std::get<0>(data);
        auto volume = std::get<1>(data);
        auto diffusionOperator = std::get<2>(data);
        
        auto outputVolume = volumeFactory->createConservedVolume(nx, 1, 1, diffusionOperator->getNumberOfGhostCells());
        outputVolume->makeZero();
        diffusionOperator->applyDiffusion(*outputVolume, *volume);

        auto outputVolumeCPU = volumeFactoryCPU->createConservedVolume(nx, 1, 1, diffusionOperator->getNumberOfGhostCells());

        outputVolume->copyTo(*outputVolumeCPU);

        double L1Norm = 0;
        volume::for_each_internal_volume_index<0>(*outputVolumeCPU, [&](size_t, size_t index, size_t) {
            for (size_t i = 0; i < outputVolumeCPU->getNumberOfVariables(); ++i) {
                
                L1Norm += std::abs(outputVolumeCPU->getScalarMemoryArea(i)->getPointer()[index]);
            }
        });

        std::cout << L1Norm << std::endl;

        errors.push_back(std::log(L1Norm/nx));
        resolutions.push_back(std::log(nx));
    }

    ASSERT_LE(parameters.expectedConvergenceRate, -linearFit(resolutions, errors)[0]);


}

INSTANTIATE_TEST_CASE_P(TecnoDiffusionTests,
    TecnoDiffusionTest,
    ::testing::Values(
        DiffusionParameters("cpu", "burgers", "tecnoroe", "none", 0.9),
        DiffusionParameters("cpu", "burgers", "tecnoroe", "eno2", 1.9),
        DiffusionParameters("cpu", "burgers", "tecnoroe", "eno3", 2.9)
        //DiffusionParameters("cuda", "burgers", "tecnoroe", "none", 0.9)
       // DiffusionParameters("cpu", "burgers", "tecnoroe", "eno2", 1.9),
        //DiffusionParameters("cpu", "burgers", "tecnoroe", "eno3", 2.9)
        ));
