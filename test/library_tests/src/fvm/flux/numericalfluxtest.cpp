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

#include <gtest/gtest.h>

#include "alsfvm/types.hpp"
#include "alsfvm/numflux/NumericalFluxFactory.hpp"
#include "alsfvm/numflux/NumericalFluxCPU.hpp"
#include "alsfvm/numflux/euler/HLL.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/equation/euler/AllVariables.hpp"
#include "alsfvm/equation/CellComputerFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/boundary/BoundaryFactory.hpp"

using namespace alsfvm::numflux;
using namespace alsfvm;
using namespace alsfvm::volume;

class NumericalFluxTest : public ::testing::Test {
public:
    std::string equation;
    std::string flux;
    std::string reconstruction;
    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration;
    alsfvm::shared_ptr<simulator::SimulatorParameters> simulatorParameters;
    NumericalFluxFactory fluxFactory;
    grid::Grid grid;
    alsfvm::shared_ptr<memory::MemoryFactory> memoryFactory;
    volume::VolumeFactory volumeFactory;

    const size_t nx;
    const size_t ny;
    const size_t nz;

    NumericalFluxTest()
        : equation("euler3"), flux("HLL"), reconstruction("none"),
          deviceConfiguration(new DeviceConfiguration("cpu")),
          simulatorParameters(new simulator::SimulatorParameters(equation, "cpu")),
          fluxFactory(equation, flux, reconstruction, simulatorParameters,
              deviceConfiguration),
          grid(rvec3(0, 0, 0), rvec3(1, 1, 1), ivec3(20, 20, 20)),
          memoryFactory(new memory::MemoryFactory(deviceConfiguration)),
          volumeFactory(equation, memoryFactory), nx(10), ny(10), nz(10) {

    }
};

TEST_F(NumericalFluxTest, ConstructionTest) {
    auto numericalFlux = fluxFactory.createNumericalFlux(grid);
}

TEST_F(NumericalFluxTest, ConsistencyTest) {
    // This test that the flux is consistent

    auto conservedVariables = volumeFactory.createConservedVolume(nx, ny, nz, 1);
    auto extraVariables = volumeFactory.createExtraVolume(nx, ny, nz, 1);


    boundary::BoundaryFactory boundaryFactory("neumann", deviceConfiguration);

    auto boundary = boundaryFactory.createBoundary(1);

    volume::fill_volume<equation::euler::ConservedVariables<3>>(*conservedVariables,
            grid,
    [](real x, real y, real z, equation::euler::ConservedVariables<3>& out) {
        out.rho = 1;
        out.m.x = 1;
        out.m.y = 1;
        out.m.z = 1;
        out.E = 10;

    });

    boundary->applyBoundaryConditions(*conservedVariables, grid);
    equation::CellComputerFactory cellComputerFactory(simulatorParameters,
        deviceConfiguration);

    auto computer = cellComputerFactory.createComputer();
    computer->computeExtraVariables(*conservedVariables, *extraVariables);

    ASSERT_TRUE(computer->obeysConstraints(*conservedVariables));
    auto output = volumeFactory.createConservedVolume(nx, ny, nz, 1);

    for (size_t i = 0; i < output->getNumberOfVariables(); i++) {
        for (size_t j = 0; j < nx * ny * nz; j++) {
            output->getScalarMemoryArea(i)->getPointer()[j] = 1;
        }
    }

    auto numericalFlux = fluxFactory.createNumericalFlux(grid);

    rvec3 waveSpeeds(0, 0, 0);
    numericalFlux->computeFlux(*conservedVariables, waveSpeeds, true, *output);

    // Check that output is what we expect
    // Here the flux should be consistent, so we expect that
    // the difference f(U,Ur)-f(Ul,U) should be zero everywhere.
    for (size_t n = 0; n < output->getNumberOfVariables(); n++) {

        for (size_t k = 1; k < nz - 1; k++) {
            for (size_t j = 1; j < ny - 1; j++) {
                for (size_t i = 1; i < nx - 1; i++) {


                    ASSERT_EQ(0, output->getScalarMemoryArea(n)->getView().at(i, j, k))
                            << "Consistency check failed at (" << i << ", " << j << ", " << k << ")";
                }
            }

        }
    }

}
