#include <gtest/gtest.h>
#include "alsfvm/types.hpp"
#include "alsfvm/diffusion/DiffusionFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "utils/polyfit.hpp"
#include "alsfvm/boundary/BoundaryFactory.hpp"
#include <iostream>
#include "alsfvm/equation/CellComputerFactory.hpp"
#include "alsfvm/equation/EquationParameterFactory.hpp"

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
      #ifndef ALSVINN_HAVE_CUDA
      this->platform = "cpu";
#endif
        
    }

   
};

std::ostream& operator<<(std::ostream& os, const DiffusionParameters& parameters) {
    os << "\n{\n\texpectedConvergenceRate = " << parameters.expectedConvergenceRate

        << "\n\tdiffusion = " << parameters.diffusion
        << "\n\tequation = " << parameters.equation
        << "\n\treconstruction = " << parameters.reconstruction
        << "\n\tplatform = " << parameters.platform << std::endl << "}" << std::endl;
    return os;
}
class TecnoDiffusionTest : public ::testing::TestWithParam <DiffusionParameters> {
public:

    TecnoDiffusionTest()
     :   parameters(GetParam())
    {
        simulatorParameters.reset(new simulator::SimulatorParameters);
        simulatorParameters->setEquationName(parameters.equation);
        simulatorParameters->setPlatform(parameters.platform);
        deviceConfiguration.reset(new DeviceConfiguration(parameters.platform));
        deviceConfigurationCPU.reset(new DeviceConfiguration("cpu"));

        memoryFactory.reset(new memory::MemoryFactory(deviceConfiguration));
        memoryFactoryCPU.reset(new memory::MemoryFactory(deviceConfigurationCPU));

        volumeFactory.reset(new volume::VolumeFactory(parameters.equation, memoryFactory));
        volumeFactoryCPU.reset(new volume::VolumeFactory(parameters.equation, memoryFactoryCPU));

        boundaryFactory.reset(new boundary::BoundaryFactory("periodic", deviceConfiguration));
        equation::EquationParameterFactory equationParameterFactory;

        auto equationParameters = equationParameterFactory.createDefaultEquationParameters(parameters.equation);
        simulatorParameters->setEquationParameters( equationParameters );
        equation::CellComputerFactory cellComputerFactory(simulatorParameters, deviceConfiguration);

        cellComputer = cellComputerFactory.createComputer();
    }

    DiffusionParameters parameters;

    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration;
    alsfvm::shared_ptr<DeviceConfiguration> deviceConfigurationCPU;

    alsfvm::shared_ptr<memory::MemoryFactory> memoryFactory;
    alsfvm::shared_ptr<memory::MemoryFactory> memoryFactoryCPU;

    alsfvm::shared_ptr<volume::VolumeFactory> volumeFactory;
    alsfvm::shared_ptr<volume::VolumeFactory> volumeFactoryCPU;

    alsfvm::shared_ptr<alsfvm::simulator::SimulatorParameters> simulatorParameters;

    alsfvm::shared_ptr<boundary::BoundaryFactory> boundaryFactory;
    alsfvm::shared_ptr<alsfvm::equation::CellComputer> cellComputer;
    std::tuple<grid::Grid, alsfvm::shared_ptr<volume::Volume>, alsfvm::shared_ptr<diffusion::DiffusionOperator> > makeSetup(size_t nx) {
        auto grid = grid::Grid(rvec3(0., 0., 0.), rvec3(1., 0., 0.), ivec3(nx, 1, 1));
        
        diffusion::DiffusionFactory diffusionFactory;

        auto diffusionOperator = diffusionFactory.createDiffusionOperator(parameters.equation, parameters.diffusion,
            parameters.reconstruction, grid, *simulatorParameters,deviceConfiguration, memoryFactory,
            *volumeFactory);

        auto boundary = boundaryFactory->createBoundary(diffusionOperator->getNumberOfGhostCells());

        auto volumeCPU = volumeFactoryCPU->createConservedVolume(nx, 1, 1, diffusionOperator->getNumberOfGhostCells());
        volumeCPU->makeZero();
        

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

            if (parameters.equation == "euler1" || parameters.equation == "euler2" || parameters.equation == "euler3") {
                // make sure the energy is compatible
                volumeCPU->getScalarMemoryArea("E")->getPointer()[index] = averageIntegralF(a, b) + 20;
            }
        });

        

        auto volume = volumeFactory->createConservedVolume(nx, 1, 1, diffusionOperator->getNumberOfGhostCells());

        volumeCPU->copyTo(*volume);
        auto extraVolume = volumeFactory->createExtraVolume(nx, 1, 1, diffusionOperator->getNumberOfGhostCells());
        cellComputer->computeExtraVariables(*volume, *extraVolume);
        //ASSERT_EQ(true, cellComputer->obeysConstraints(*volume, *extraVolume));
        boundary->applyBoundaryConditions(*volume, grid);
        return std::make_tuple(grid, volume, diffusionOperator);
    }
};


TEST_P(TecnoDiffusionTest, OrderTest) {

    // we check for each component, to make sure none
    // of the components are always zero eg.
    std::vector<std::vector<real>> errors;
    std::vector<std::vector<real>> resolutions;

    const int minK = 5;
    const int maxK = 14;

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

        if (errors.size() == 0) {
            errors.resize(outputVolume->getNumberOfVariables());
            resolutions.resize(outputVolume->getNumberOfVariables());
        }

        for (size_t i = 0; i < outputVolumeCPU->getNumberOfVariables(); ++i) {
           
            double L1Norm = 0;
            volume::for_each_internal_volume_index<0>(*outputVolumeCPU, [&](size_t, size_t index, size_t) {
                L1Norm += std::abs(outputVolumeCPU->getScalarMemoryArea(i)->getPointer()[index]);

            });


            if (i == 0) {
                std::cout << "[" << nx ;
            }
            std::cout << ", ["<< L1Norm / nx;
            if (k > minK) {
                auto rate = (std::log(L1Norm/nx) - errors[i].back())/(std::log(real(nx))-resolutions[i].back());
                std::cout << ", " << rate;
            }
            std::cout << "]";

            if (i == outputVolumeCPU->getNumberOfVariables() - 1) {
                std::cout << "]\n";
            }

            errors[i].push_back(std::log(L1Norm / nx));
            resolutions[i].push_back(std::log(real(nx)));


        }
    }

    for (size_t i = 0; i < errors.size(); ++i) {
       
        ASSERT_LE(parameters.expectedConvergenceRate, -linearFit(resolutions[i], errors[i])[0]);
    }


}


#if 1
INSTANTIATE_TEST_CASE_P(TecnoDiffusionTestsTecnoeRoe,
    TecnoDiffusionTest,
    ::testing::Values(
        DiffusionParameters("cpu", "burgers", "tecnoroe", "none", 1.9),
        DiffusionParameters("cpu", "burgers", "tecnoroe", "eno2", 2.9),
        DiffusionParameters("cpu", "burgers", "tecnoroe", "eno3", 3.8),
        DiffusionParameters("cpu", "burgers", "tecnoroe", "eno4", 3.9),
        DiffusionParameters("cuda", "burgers", "tecnoroe", "none", 1.9),
        DiffusionParameters("cuda", "burgers", "tecnoroe", "eno2", 2.9),
        DiffusionParameters("cuda", "burgers", "tecnoroe", "eno3", 3.8),
        DiffusionParameters("cuda", "burgers", "tecnoroe", "eno4", 3.9),
       DiffusionParameters("cpu", "euler1", "tecnoroe", "none", 1.9),
       DiffusionParameters("cpu", "euler1", "tecnoroe", "eno2", 2.9),
       DiffusionParameters("cpu", "euler1", "tecnoroe", "eno3", 3.8),
       DiffusionParameters("cpu", "euler1", "tecnoroe", "eno4", 3.5),
       DiffusionParameters("cuda", "euler1", "tecnoroe", "none", 1.9),
       DiffusionParameters("cuda", "euler1", "tecnoroe", "eno2", 2.9),
       DiffusionParameters("cuda", "euler1", "tecnoroe", "eno3", 3.8),
       DiffusionParameters("cuda", "euler1", "tecnoroe", "eno4", 3.5)
       //DiffusionParameters("cpu", "euler2", "tecnoroe", "none", 1.9),
       //DiffusionParameters("cpu", "euler2", "tecnoroe", "eno2", 2.9),
       //DiffusionParameters("cpu", "euler2", "tecnoroe", "eno3", 3.8),
       //DiffusionParameters("cpu", "euler2", "tecnoroe", "eno4", 3.5),
       //DiffusionParameters("cuda", "euler2", "tecnoroe", "none", 1.9),
       //DiffusionParameters("cuda", "euler2", "tecnoroe", "eno2", 2.9),
       //DiffusionParameters("cuda", "euler2", "tecnoroe", "eno3", 3.8),
       //DiffusionParameters("cuda", "euler2", "tecnoroe", "eno4", 3.5),
       //DiffusionParameters("cpu", "euler3", "tecnoroe", "none", 1.9),
       //DiffusionParameters("cpu", "euler3", "tecnoroe", "eno2", 2.9),
       //DiffusionParameters("cpu", "euler3", "tecnoroe", "eno3", 3.8),
       //DiffusionParameters("cpu", "euler3", "tecnoroe", "eno4", 3.5),
       //DiffusionParameters("cuda", "euler3", "tecnoroe", "none", 1.9),
       //DiffusionParameters("cuda", "euler3", "tecnoroe", "eno2", 2.9),
       //DiffusionParameters("cuda", "euler3", "tecnoroe", "eno3", 3.8),
       //DiffusionParameters("cuda", "euler3", "tecnoroe", "eno4", 3.9)

        ));

INSTANTIATE_TEST_CASE_P(TecnoDiffusionTestsTecnoeRusanov,
    TecnoDiffusionTest,
    ::testing::Values(
        DiffusionParameters("cpu", "burgers",  "tecnorusanov", "none", 1.9),
        DiffusionParameters("cpu", "burgers",  "tecnorusanov", "eno2", 2.9),
        DiffusionParameters("cpu", "burgers",  "tecnorusanov", "eno3", 3.8),
        DiffusionParameters("cpu", "burgers",  "tecnorusanov", "eno4", 3.9),
        DiffusionParameters("cuda", "burgers", "tecnorusanov", "none", 1.9),
        DiffusionParameters("cuda", "burgers", "tecnorusanov", "eno2", 2.9),
        DiffusionParameters("cuda", "burgers", "tecnorusanov", "eno3", 3.8),
        DiffusionParameters("cuda", "burgers", "tecnorusanov", "eno4", 3.9)
       //DiffusionParameters("cpu",  "euler1",  "tecnorusanov", "none", 1.9),
       //DiffusionParameters("cpu",  "euler1",  "tecnorusanov", "eno2", 2.9),
       //DiffusionParameters("cpu",  "euler1",  "tecnorusanov", "eno3", 3.8),
       //DiffusionParameters("cpu",  "euler1",  "tecnorusanov", "eno4", 3.5),
       //DiffusionParameters("cuda", "euler1",  "tecnorusanov", "none", 1.9),
       //DiffusionParameters("cuda", "euler1",  "tecnorusanov", "eno2", 2.9),
       //DiffusionParameters("cuda", "euler1",  "tecnorusanov", "eno3", 3.8),
       //DiffusionParameters("cuda", "euler1",  "tecnorusanov", "eno4", 3.5),
       //DiffusionParameters("cpu",  "euler2",  "tecnorusanov", "none", 1.9),
       //DiffusionParameters("cpu",  "euler2",  "tecnorusanov", "eno2", 2.9),
       //DiffusionParameters("cpu",  "euler2",  "tecnorusanov", "eno3", 3.8),
       //DiffusionParameters("cpu",  "euler2",  "tecnorusanov", "eno4", 3.5),
       //DiffusionParameters("cuda", "euler2",  "tecnorusanov", "none", 1.9),
       //DiffusionParameters("cuda", "euler2",  "tecnorusanov", "eno2", 2.9),
       //DiffusionParameters("cuda", "euler2",  "tecnorusanov", "eno3", 3.8),
       //DiffusionParameters("cuda", "euler2",  "tecnorusanov", "eno4", 3.5),
       //DiffusionParameters("cpu",  "euler3",  "tecnorusanov", "none", 1.9),
       //DiffusionParameters("cpu",  "euler3",  "tecnorusanov", "eno2", 2.9),
       //DiffusionParameters("cpu",  "euler3",  "tecnorusanov", "eno3", 3.8),
       //DiffusionParameters("cpu",  "euler3",  "tecnorusanov", "eno4", 3.5),
       //DiffusionParameters("cuda", "euler3",  "tecnorusanov", "none", 1.9),
       //DiffusionParameters("cuda", "euler3",  "tecnorusanov", "eno2", 2.9),
       //DiffusionParameters("cuda", "euler3",  "tecnorusanov", "eno3", 3.8),
       //DiffusionParameters("cuda", "euler3",  "tecnorusanov", "eno4", 3.9)
        ));
#else
//INSTANTIATE_TEST_CASE_P(TecnoDiffusionTestsTecnoeRusanov,
//    TecnoDiffusionTest,
//    ::testing::Values(
//        DiffusionParameters("cuda", "euler2",  "tecnorusanov", "none", 1.9),
//                            DiffusionParameters("cuda", "euler2", "tecnorusanov", "none", 1.9)
//                            //DiffusionParameters("cuda", "euler1", "tecnorusanov", "eno2", 2.9),
//                            //DiffusionParameters("cuda", "euler1", "tecnorusanov", "eno3", 3.8),
//                            //DiffusionParameters("cuda", "euler1", "tecnorusanov", "eno4", 3.9)
//                            ));
#endif
