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

#include "alsfvm/equation/CellComputerFactory.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/equation/euler/AllVariables.hpp"
#include "alsfvm/equation/euler/Euler.hpp"


using namespace alsfvm;
using namespace alsfvm::memory;
using namespace alsfvm::equation;
using namespace alsfvm::volume;

class TestExtraComputation : public ::testing::Test {
public:
    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration;
    const std::string equation;
    const std::string platform;
    const size_t nx;
    const size_t ny;
    const size_t nz;
    alsfvm::shared_ptr<MemoryFactory> memoryFactory;
    VolumeFactory volumeFactory;
    alsfvm::shared_ptr<Volume> conservedVolume;
    alsfvm::shared_ptr<Volume> extraVolume;
    alsfvm::shared_ptr<simulator::SimulatorParameters> simulatorParameters;
    CellComputerFactory cellComputerFactory;
    TestExtraComputation()
        : deviceConfiguration(new DeviceConfiguration("cpu")),
          equation("euler3"),
          platform("cpu"),
          nx(10), ny(10), nz(10),
          memoryFactory(new MemoryFactory(deviceConfiguration)),
          volumeFactory("euler3", memoryFactory),
          conservedVolume(volumeFactory.createConservedVolume(nx, ny, nz)),
          extraVolume(volumeFactory.createExtraVolume(nx, ny, nz)),
          simulatorParameters(new simulator::SimulatorParameters("euler3", "cpu")),
          cellComputerFactory(simulatorParameters, deviceConfiguration) {

    }
};

TEST_F(TestExtraComputation, ConstructTest) {
    auto cellComputer = cellComputerFactory.createComputer();
}

TEST_F(TestExtraComputation, CheckExtraCalculation) {
    auto cellComputer = cellComputerFactory.createComputer();

    // Fill up volume
    transform_volume<euler::ConservedVariables<3>, euler::ConservedVariables<3>>
        (*conservedVolume, *conservedVolume, [](const euler::ConservedVariables<3>& in)
    -> euler::ConservedVariables<3> {
        return euler::ConservedVariables<3>(0.5, rvec3{ 1, 1, 1 }, 4.4);
    });

    cellComputer->computeExtraVariables(*conservedVolume, *extraVolume);

    auto& eulerParameters = static_cast<euler::EulerParameters&>
        (simulatorParameters->getEquationParameters());
    for_each_cell<euler::ExtraVariables<3>>(*extraVolume, [&](
    const euler::ExtraVariables<3>& in, size_t index) {
        ASSERT_EQ(in.u, rvec3(2, 2, 2));
        ASSERT_EQ(in.p, (eulerParameters.getGamma() - 1) * (4.4 - 0.5 * 3 / 0.5));
    });

}

TEST_F(TestExtraComputation, CheckMaximumWaveSpeed) {
    auto cellComputer = cellComputerFactory.createComputer();
    auto& eulerParameters = static_cast<euler::EulerParameters&>
        (simulatorParameters->getEquationParameters());
    // Fill up volume
    transform_volume<euler::ConservedVariables<3>, euler::ConservedVariables<3>>
        (*conservedVolume, *conservedVolume, [](const euler::ConservedVariables<3>& in)
    -> euler::ConservedVariables<3> {
        return euler::ConservedVariables<3>(0.5, rvec3{ 1, 1, 1 }, 4.4);
    });

    const real gamma = eulerParameters.getGamma();
    cellComputer->computeExtraVariables(*conservedVolume, *extraVolume);
    {
        real maxWaveSpeed = cellComputer->computeMaxWaveSpeed(*conservedVolume,
                *extraVolume, 0);

        ASSERT_EQ(maxWaveSpeed, 2 + sqrt(gamma * (gamma - 1) * (4.4 - 0.5 * 3 / 0.5) /
                0.5));
    }

    {
        real maxWaveSpeed = cellComputer->computeMaxWaveSpeed(*conservedVolume,
                *extraVolume, 1);

        ASSERT_EQ(maxWaveSpeed, 2 + sqrt(gamma * (gamma - 1) * (4.4 - 0.5 * 3 / 0.5) /
                0.5));
    }

    {
        real maxWaveSpeed = cellComputer->computeMaxWaveSpeed(*conservedVolume,
                *extraVolume, 2);

        ASSERT_EQ(maxWaveSpeed, 2 + sqrt(gamma * (gamma - 1) * (4.4 - 0.5 * 3 / 0.5) /
                0.5));
    }
}


TEST_F(TestExtraComputation, CheckConstraints) {
    auto cellComputer = cellComputerFactory.createComputer();

    // Fill up volume
    transform_volume<euler::ConservedVariables<3>, euler::ConservedVariables<3>>
        (*conservedVolume, *conservedVolume, [](const euler::ConservedVariables<3>& in)
    -> euler::ConservedVariables<3> {
        return euler::ConservedVariables<3>(0.5, rvec3{ 1, 1, 1 }, 4.4);
    });

    cellComputer->computeExtraVariables(*conservedVolume, *extraVolume);

    // This should be fine
    ASSERT_TRUE(cellComputer->obeysConstraints(*conservedVolume, *extraVolume));

    // Now we fill it with something that cancels the constraints
    conservedVolume->getScalarMemoryArea("rho")->getPointer()[4] = -0.4;

    ASSERT_FALSE(cellComputer->obeysConstraints(*conservedVolume, *extraVolume));

    // and some extra
    extraVolume->getScalarMemoryArea("p")->getPointer()[8] = -0.4;
    ASSERT_FALSE(cellComputer->obeysConstraints(*conservedVolume, *extraVolume));

    // And fix the first one, then we should still get something false
    conservedVolume->getScalarMemoryArea("rho")->getPointer()[4] = 2;
    ASSERT_FALSE(cellComputer->obeysConstraints(*conservedVolume, *extraVolume));

    // check that it can be fixed again
    extraVolume->getScalarMemoryArea("p")->getPointer()[8] = 0.4;
    ASSERT_TRUE(cellComputer->obeysConstraints(*conservedVolume, *extraVolume));


    // Add an inf
    extraVolume->getScalarMemoryArea("p")->getPointer()[8] = INFINITY;
    ASSERT_FALSE(cellComputer->obeysConstraints(*conservedVolume, *extraVolume));

}
