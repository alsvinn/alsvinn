/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

// The main goal of this test is to see if the different components work together
// and that they can simulate some known test cases without causing obvious errors
// (obvious errors being defined as nan-errors or out of bounds error wrt. to the equation,
// ie. check if rho>0).

#include <gtest/gtest.h>
#include "alsfvm/grid/Grid.hpp"
#include "alsfvm/memory/MemoryFactory.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/numflux/NumericalFluxFactory.hpp"
#include "alsfvm/equation/CellComputerFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/integrator/IntegratorFactory.hpp"
#include "alsfvm/io/HDF5Writer.hpp"
#include "alsfvm/boundary/BoundaryFactory.hpp"
#include <array>
#include "alsfvm/reconstruction/ENOCPU.hpp"
#include "alsfvm/simulator/ConservedSystem.hpp"
#include "alsfvm/reconstruction/WENOCPU.hpp"
#include "alsfvm/diffusion/NoDiffusion.hpp"
#include <limits>
#include <iomanip>

using namespace alsfvm;
using namespace alsfvm::grid;
using namespace alsfvm::memory;
using namespace alsfvm::volume;
using namespace alsfvm::numflux;
using namespace alsfvm::equation::euler;
using namespace alsfvm::equation;
using namespace alsfvm::boundary;
void runTest(
    std::function<void(real x, real y, real z, ConservedVariables<2>& u, ExtraVariables<2>& v)>
    initialData, size_t N,
    const std::string& reconstruction, const real T, const std::string& name) {
    const real cfl = reconstruction.find("eno") != reconstruction.npos ? 0.475 :
        0.9;
    std::cout << "using cfl = " << cfl << std::endl;

    const std::string integratorName = (reconstruction.find("eno") !=
            reconstruction.npos ? "rungekutta2" : "forwardeuler");


    std::cout << "Using integrator " << integratorName << std::endl;
    const size_t numberOfSaves = 10;

    const real saveInterval = T / numberOfSaves;
    Grid grid(rvec3(0, 0, 0), rvec3(1, 1, 1), ivec3(N, N, 1));

    auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>("cpu");

    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("euler2", memoryFactory);

    auto simulatorParameters =
        alsfvm::make_shared<simulator::SimulatorParameters>("euler2", "cpu");
    NumericalFluxFactory fluxFactory("euler2", "HLL", reconstruction,
        simulatorParameters, deviceConfiguration);
    CellComputerFactory cellComputerFactory(simulatorParameters,
        deviceConfiguration);
    BoundaryFactory boundaryFactory("neumann", deviceConfiguration);




    auto numericalFlux = fluxFactory.createNumericalFlux(grid);
    alsfvm::shared_ptr<integrator::System> system(new simulator::ConservedSystem(
            numericalFlux,
            alsfvm::dynamic_pointer_cast<diffusion::DiffusionOperator>
            (alsfvm::make_shared<diffusion::NoDiffusion>())));
    integrator::IntegratorFactory integratorFactory(integratorName);
    auto integrator = integratorFactory.createIntegrator(system);

    std::vector<alsfvm::shared_ptr<volume::Volume> > conservedVolumes(
        integrator->getNumberOfSubsteps() + 1);

    for (size_t i = 0; i < conservedVolumes.size(); i++) {
        conservedVolumes[i] = volumeFactory.createConservedVolume(N, N, 1,
                numericalFlux->getNumberOfGhostCells());
    }

    auto extra1 = volumeFactory.createExtraVolume(N, N, 1,
            numericalFlux->getNumberOfGhostCells());
    auto extra2 = volumeFactory.createExtraVolume(N, N, 1,
            numericalFlux->getNumberOfGhostCells());


    fill_volume<ConservedVariables<2>, ExtraVariables<2>>(*conservedVolumes[0],
            *extra1, grid,
            initialData);


    auto cellComputer = cellComputerFactory.createComputer();

    real t = 0;

    io::HDF5Writer writer(name);
    writer.write(*conservedVolumes[0], grid,
        simulator::TimestepInformation());


    int i = 0;
    size_t numberOfTimesteps = 0;
    auto boundary = boundaryFactory.createBoundary(
            numericalFlux->getNumberOfGhostCells());
    size_t nsaves = 1;

    boundary->applyBoundaryConditions(*conservedVolumes[0], grid);
    cellComputer->computeExtraVariables(*conservedVolumes[0], *extra1);
    ASSERT_TRUE(cellComputer->obeysConstraints(*conservedVolumes[0]));

    writer.write(*conservedVolumes[0], grid,
        simulator::TimestepInformation());
    simulator::TimestepInformation timestepInformation;

    while (timestepInformation.getCurrentTime() < T) {
        real dt = 0;
        numberOfTimesteps++;

        for (size_t substep = 0; substep < integrator->getNumberOfSubsteps();
            substep++) {
            auto conservedNext = conservedVolumes[substep + 1];

            dt = integrator->performSubstep(conservedVolumes, grid.getCellLengths(), dt,
                    cfl, *conservedNext, substep, timestepInformation);

            std::array<real*, 5> conservedPointers = {{
                    conservedNext->getScalarMemoryArea(0)->getPointer(),
                    conservedNext->getScalarMemoryArea(1)->getPointer(),
                    conservedNext->getScalarMemoryArea(2)->getPointer(),
                    conservedNext->getScalarMemoryArea(3)->getPointer(),
                    conservedNext->getScalarMemoryArea(4)->getPointer()
                }
            };

            std::array<real*, 4> extraPointers = {{
                    extra1->getScalarMemoryArea(0)->getPointer(),
                    extra1->getScalarMemoryArea(1)->getPointer(),
                    extra1->getScalarMemoryArea(2)->getPointer(),
                    extra1->getScalarMemoryArea(3)->getPointer()

                }
            };

            boundary->applyBoundaryConditions(*conservedNext, grid);

            cellComputer->computeExtraVariables(*conservedNext, *extra1);

            // Intense error checking below. We basically check that the output is sane,
            // this doubles the work of cellComputer->obeysConstraints.
            for (size_t i = 0; i < conservedNext->getScalarMemoryArea(0)->getSize(); i++) {
                // Check that density and pressure is positive
                ASSERT_GT(conservedPointers[0][i], 0);
                ASSERT_GT(extraPointers[0][i], 0);

                for (size_t j = 0; j < 5; j++) {
                    if (std::isnan(conservedPointers[j][i])) {
                        std::cout << "(conserved) nan at i = " << i;
                        std::cout << " variable " << conservedVolumes[0]->getName(j) << std::endl;
                    }

                    ASSERT_FALSE(std::isnan(conservedPointers[j][i]));

                    if (std::isinf(conservedPointers[j][i])) {
                        std::cout << "(conserved) inf at i = " << i;
                        std::cout << " variable " << conservedVolumes[0]->getName(j) << std::endl;
                    }

                    ASSERT_FALSE(std::isinf(conservedPointers[j][i]));
                }

                for (size_t j = 0; j < 4; j++) {
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



            ASSERT_TRUE(cellComputer->obeysConstraints(*conservedNext));



        }

        conservedVolumes[0].swap(conservedVolumes.back());
        ASSERT_FALSE(std::isnan(dt));

        ASSERT_GT(dt, 0);
        i++;

        if (t >= nsaves * saveInterval) {
            nsaves++;

            std::cout << "saving at t " << t << " (nsaves * saveInterval = " << nsaves*
                saveInterval << ")" << std::endl;
            writer.write(*conservedVolumes[0], grid,
                simulator::TimestepInformation());
        }

        timestepInformation.incrementTime(dt);


    }

    std::cout << "Number of timesteps used: " << numberOfTimesteps << std::endl;
    writer.write(*conservedVolumes[0], grid,
        simulator::TimestepInformation());
}
TEST(EulerTest, ShockTubeTest) {
    equation::euler::EulerParameters parameters;
    runTest([&](real x, real y, real z, ConservedVariables<2>& u,
    ExtraVariables<2>& v) {

        if (x < 0.04) {
            u.rho = 3.86859;
            v.u.x = 11.2536;
            v.p = 167.345;
        } else {
            real r = pow(x - 0.25, 2) + pow(y - 0.5, 2);
            real r_max = pow(0.15, 2);

            if (r <= r_max) {
                u.rho = 10.0;
            } else {
                u.rho = 1.0;
            }

            v.p = 1.0;
        }

        u.m = u.rho * v.u;
        u.E = v.p / (parameters.getGamma() - 1) + 0.5 * u.rho * v.u.dot(v.u);
    }, 128, "none", 0.06, "euler_shocktube");
}

TEST(EulerTest, ShockVortex) {
    equation::euler::EulerParameters parameters;
    runTest([&](real x, real y, real z, ConservedVariables<2>& u,
    ExtraVariables<2>& v) {
        real epsilon = 0.3, r_c = 0.05, alpha = 0.204, x_c = 0.25, y_c = 0.5;

        // shock part
        if (x < 0.5) {
            u.rho = 1.0;
            v.u.x = sqrt(parameters.getGamma());
            v.u.y = 0.0;
            v.p = 1.0;
        } else {
            u.rho = 1.0 / 1.1;
            v.u.x = 1.1 * sqrt(parameters.getGamma());
            v.u.y = 0.0;
            v.p = 1 - 0.1 * parameters.getGamma();
        }

        // vortex part
        if (x < 0.5) {
            real tau = sqrt(pow(x - x_c, 2) + pow(y - y_c, 2)) / r_c;
            real sin_theta = (y - y_c) / (tau * r_c);
            real cos_theta = (x - x_c) / (tau * r_c);

            v.u.x += epsilon * tau * exp(alpha * (1 - pow(tau, 2))) * sin_theta;
            v.u.y += -epsilon * tau * exp(alpha * (1 - pow(tau, 2))) * cos_theta;
            v.p += -(parameters.getGamma() - 1) * pow(epsilon,
                    2) * exp(2 * alpha * (1 - pow(tau,
                            2))) / (4 * alpha * parameters.getGamma()) * u.rho;
        }

        u.m = u.rho * v.u;
        u.E = v.p / (parameters.getGamma() - 1) + 0.5 * u.rho * v.u.dot(v.u);

    }, 256, "eno2", 0.35, "euler_vortex");
}
