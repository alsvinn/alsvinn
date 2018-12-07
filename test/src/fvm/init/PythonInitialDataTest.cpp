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

#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/init/PythonInitialData.hpp"
#include "alsfvm/equation/CellComputerFactory.hpp"

using namespace alsfvm;
using namespace alsfvm::memory;
using namespace alsfvm::volume;
using namespace alsfvm::grid;
using namespace alsfvm::init;

TEST(PythonInitialDataTest, AnswerToEverything) {


    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration(
        new DeviceConfiguration);
    auto simulatorParameters =
        alsfvm::make_shared<simulator::SimulatorParameters>("euler3", "cpu");
    equation::CellComputerFactory cellComputerFactory(simulatorParameters,
        deviceConfiguration);
    auto cellComputer = cellComputerFactory.createComputer();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);
    volume::VolumeFactory volumeFactory("euler3", memoryFactory);
    size_t nx = 10;
    size_t ny = 10;
    size_t nz = 1;

    grid::Grid grid({0., 0., 0.}, {1, 1, 1}, {int(nx), int(ny), int(nz)});


    auto volumeConserved = volumeFactory.createConservedVolume(nx, ny, nz, 1);
    auto volumeExtra = volumeFactory.createExtraVolume(nx, ny, nz, 1);
    auto volumePrimitive = volumeFactory.createPrimitiveVolume(nx, ny, nz, 1);

    // Fill every variable with 42
    const std::string pythonCode = "rho = 42\nux=42\nuy=42\nuz=42\np=42";

    PythonInitialData initialData(pythonCode, Parameters());

    initialData.setInitialData(*volumeConserved,
        *volumeExtra,
        *volumePrimitive,
        *cellComputer,
        grid);

    volume::for_each_midpoint(*volumePrimitive, grid, [&](real x, real y, real z,
    size_t index) {
        for (size_t var = 0; var < volumePrimitive->getNumberOfVariables(); ++var) {
            ASSERT_EQ(42, volumePrimitive->getScalarMemoryArea(var)->getPointer()[index]);
        }
    });

}

TEST(PythonInitialDataTest, ParameterTest) {


    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration(
        new DeviceConfiguration);
    auto simulatorParameters =
        alsfvm::make_shared<simulator::SimulatorParameters>("euler3", "cpu");
    equation::CellComputerFactory cellComputerFactory(simulatorParameters,
        deviceConfiguration);
    auto cellComputer = cellComputerFactory.createComputer();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);
    volume::VolumeFactory volumeFactory("euler3", memoryFactory);
    size_t nx = 10;
    size_t ny = 10;
    size_t nz = 1;

    grid::Grid grid({ 0., 0., 0. }, { 1, 1, 1 }, { int(nx), int(ny), int(nz) });


    auto volumeConserved = volumeFactory.createConservedVolume(nx, ny, nz, 1);
    auto volumeExtra = volumeFactory.createExtraVolume(nx, ny, nz, 1);
    auto volumePrimitive = volumeFactory.createPrimitiveVolume(nx, ny, nz, 1);
    Parameters parameters;
    parameters.addParameter("nonVector", { 43. });
    parameters.addParameter("vectorParameter", { 44., 45., 45.5, 46 });
    // Fill every variable with 42
    const std::string pythonCode =
        "rho = nonVector\nux=vectorParameter[0]\nuy=vectorParameter[1]\nuz=vectorParameter[2]\np=vectorParameter[3]+len(vectorParameter)";

    PythonInitialData initialData(pythonCode, parameters);

    initialData.setInitialData(*volumeConserved,
        *volumeExtra,
        *volumePrimitive,
        *cellComputer,
        grid);

    volume::for_each_midpoint(*volumePrimitive, grid, [&](real x, real y, real z,
    size_t index) {

        ASSERT_EQ(43, volumePrimitive->getScalarMemoryArea("rho")->getPointer()[index]);
        ASSERT_EQ(44, volumePrimitive->getScalarMemoryArea("ux")->getPointer()[index]);
        ASSERT_EQ(45, volumePrimitive->getScalarMemoryArea("uy")->getPointer()[index]);
        ASSERT_EQ(45.5,
            volumePrimitive->getScalarMemoryArea("uz")->getPointer()[index]);
        ASSERT_EQ(46 + 4,
            volumePrimitive->getScalarMemoryArea("p")->getPointer()[index]);

    });

}

TEST(PythonInitialDataTest, RiemannProblem) {

    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration(
        new DeviceConfiguration);
    auto simulatorParameters =
        alsfvm::make_shared<simulator::SimulatorParameters>("euler3", "cpu");
    equation::CellComputerFactory cellComputerFactory(simulatorParameters,
        deviceConfiguration);
    auto cellComputer = cellComputerFactory.createComputer();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);
    volume::VolumeFactory volumeFactory("euler3", memoryFactory);
    size_t nx = 10;
    size_t ny = 10;
    size_t nz = 1;

    grid::Grid grid({0., 0., 0.}, {1, 1, 1}, {int(nx), int(ny), int(nz)});


    auto volumeConserved = volumeFactory.createConservedVolume(nx, ny, nz, 1);
    auto volumeExtra = volumeFactory.createExtraVolume(nx, ny, nz, 1);
    auto volumePrimitive = volumeFactory.createPrimitiveVolume(nx, ny, nz, 1);

    // Fill every variable with 42
    const std::string pythonCode = "if x> 0.5:\n"
        "    rho = 42\n"
        "    ux=42\n"
        "    uy=42\n"
        "    uz=42\n"
        "    p=42\n"
        "else:\n"
        "    rho = 2\n"
        "    ux=2\n"
        "    uy=2\n"
        "    uz=2\n"
        "    p=2\n";


    PythonInitialData initialData(pythonCode, Parameters());

    initialData.setInitialData(*volumeConserved,
        *volumeExtra,
        *volumePrimitive,
        *cellComputer,
        grid);

    volume::for_each_midpoint(*volumePrimitive, grid, [&](real x, real y, real z,
    size_t index) {
        for (size_t var = 0; var < volumePrimitive->getNumberOfVariables(); ++var) {
            if (x > 0.5) {
                ASSERT_EQ(42, volumePrimitive->getScalarMemoryArea(var)->getPointer()[index]);
            } else {
                ASSERT_EQ(2, volumePrimitive->getScalarMemoryArea(var)->getPointer()[index]);
            }
        }
    });

}


TEST(PythonInitialDataTest, SineCosineExp) {

    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration(
        new DeviceConfiguration);
    auto simulatorParameters =
        alsfvm::make_shared<simulator::SimulatorParameters>("euler3", "cpu");
    equation::CellComputerFactory cellComputerFactory(simulatorParameters,
        deviceConfiguration);
    auto cellComputer = cellComputerFactory.createComputer();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);
    volume::VolumeFactory volumeFactory("euler3", memoryFactory);
    size_t nx = 10;
    size_t ny = 10;
    size_t nz = 1;

    grid::Grid grid({ 0., 0., 0. }, { 1, 1, 1 }, { int(nx), int(ny), int(nz) });


    auto volumeConserved = volumeFactory.createConservedVolume(nx, ny, nz, 1);
    auto volumeExtra = volumeFactory.createExtraVolume(nx, ny, nz, 1);
    auto volumePrimitive = volumeFactory.createPrimitiveVolume(nx, ny, nz, 1);

    // Fill every variable with 42
    const std::string pythonCode = ""
        "rho=sin(x)\n"
        "ux=cos(x)\n"
        "uy=exp(x)\n"
        "uz=x\n"
        "p=y\n";


    PythonInitialData initialData(pythonCode, Parameters());

    initialData.setInitialData(*volumeConserved,
        *volumeExtra,
        *volumePrimitive,
        *cellComputer,
        grid);

    volume::for_each_midpoint(*volumePrimitive, grid, [&](real x, real y, real z,
    size_t index) {
        ASSERT_EQ(std::sin(x),
            volumePrimitive->getScalarMemoryArea("rho")->getPointer()[index]);
        ASSERT_EQ(std::cos(x),
            volumePrimitive->getScalarMemoryArea("ux")->getPointer()[index]);
        ASSERT_EQ(std::exp(x),
            volumePrimitive->getScalarMemoryArea("uy")->getPointer()[index]);
        ASSERT_EQ(x, volumePrimitive->getScalarMemoryArea("uz")->getPointer()[index]);
        ASSERT_EQ(y, volumePrimitive->getScalarMemoryArea("p")->getPointer()[index]);

    });

}




TEST(PythonInitialDataTest, AnswerToEverythingInitGlobal) {


    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration(
        new DeviceConfiguration);
    auto simulatorParameters =
        alsfvm::make_shared<simulator::SimulatorParameters>("euler3", "cpu");
    equation::CellComputerFactory cellComputerFactory(simulatorParameters,
        deviceConfiguration);
    auto cellComputer = cellComputerFactory.createComputer();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);
    volume::VolumeFactory volumeFactory("euler3", memoryFactory);
    size_t nx = 10;
    size_t ny = 10;
    size_t nz = 1;

    grid::Grid grid({0., 0., 0.}, {1, 1, 1}, {int(nx), int(ny), int(nz)});


    auto volumeConserved = volumeFactory.createConservedVolume(nx, ny, nz, 1);
    auto volumeExtra = volumeFactory.createExtraVolume(nx, ny, nz, 1);
    auto volumePrimitive = volumeFactory.createPrimitiveVolume(nx, ny, nz, 1);

    // Fill every variable with 42
    const std::string pythonCode =
        "def init_global(rho, ux, uy, uz, p, nx, ny, nz, ax, ay, az, bx, by, bz):\n"
        "    rho[:,:,:] = ones((nx, ny, nz))*42\n"
        "    ux[:,:,:] = ones((nx, ny, nz))*42\n"
        "    uy[:,:,:] = ones((nx, ny, nz))*42\n"
        "    uz[:,:,:] = ones((nx, ny, nz))*42\n"
        "    p[:,:,:] = ones((nx, ny, nz))*42\n";
    PythonInitialData initialData(pythonCode, Parameters());

    initialData.setInitialData(*volumeConserved,
        *volumeExtra,
        *volumePrimitive,
        *cellComputer,
        grid);


    volume::for_each_midpoint(*volumePrimitive, grid, [&](real x, real y, real z,
    size_t index) {
        for (size_t var = 0; var < volumePrimitive->getNumberOfVariables(); ++var) {
            ASSERT_EQ(42, volumePrimitive->getScalarMemoryArea(var)->getPointer()[index])
                    << "Failed at index = " << index
                        << " variable = " << volumePrimitive->getName(var) << "\n"
                        << "Position = (" << x << ", " << y << ", " << z << ")\n";
        }
    });

}



TEST(PythonInitialDataTest, AnswerToEverythingInitGlobal3D) {


    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration(
        new DeviceConfiguration);
    auto simulatorParameters =
        alsfvm::make_shared<simulator::SimulatorParameters>("euler3", "cpu");
    equation::CellComputerFactory cellComputerFactory(simulatorParameters,
        deviceConfiguration);
    auto cellComputer = cellComputerFactory.createComputer();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);
    volume::VolumeFactory volumeFactory("euler3", memoryFactory);
    size_t nx = 10;
    size_t ny = 10;
    size_t nz = 10;

    grid::Grid grid({0., 0., 0.}, {1, 1, 1}, {int(nx), int(ny), int(nz)});


    auto volumeConserved = volumeFactory.createConservedVolume(nx, ny, nz, 1);
    auto volumeExtra = volumeFactory.createExtraVolume(nx, ny, nz, 1);
    auto volumePrimitive = volumeFactory.createPrimitiveVolume(nx, ny, nz, 1);

    // Fill every variable with 42
    const std::string pythonCode =
        "def init_global(rho, ux, uy, uz, p, nx, ny, nz, ax, ay, az, bx, by, bz):\n"
        "    rho[:,:,:] = ones((nx, ny, nz))*42\n"
        "    ux[:,:,:] = ones((nx, ny, nz))*42\n"
        "    uy[:,:,:] = ones((nx, ny, nz))*42\n"
        "    uz[:,:,:] = ones((nx, ny, nz))*42\n"
        "    p[:,:,:] = ones((nx, ny, nz))*42\n";
    PythonInitialData initialData(pythonCode, Parameters());

    initialData.setInitialData(*volumeConserved,
        *volumeExtra,
        *volumePrimitive,
        *cellComputer,
        grid);

    volume::for_each_midpoint(*volumePrimitive, grid, [&](real x, real y, real z,
    size_t index) {
        for (size_t var = 0; var < volumePrimitive->getNumberOfVariables(); ++var) {
            ASSERT_EQ(42, volumePrimitive->getScalarMemoryArea(var)->getPointer()[index])
                    << "Failed at index = " << index
                        << " variable = " << volumePrimitive->getName(var) << "\n"
                        << "Position = (" << x << ", " << y << ", " << z << ")\n";
        }
    });

}


TEST(PythonInitialDataTest, Index3D) {


    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration(
        new DeviceConfiguration);
    auto simulatorParameters =
        alsfvm::make_shared<simulator::SimulatorParameters>("euler3", "cpu");
    equation::CellComputerFactory cellComputerFactory(simulatorParameters,
        deviceConfiguration);
    auto cellComputer = cellComputerFactory.createComputer();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);
    volume::VolumeFactory volumeFactory("euler3", memoryFactory);
    size_t nx = 10;
    size_t ny = 10;
    size_t nz = 10;

    grid::Grid grid({0., 0., 0.}, {1, 1, 1}, {int(nx), int(ny), int(nz)});


    auto volumeConserved = volumeFactory.createConservedVolume(nx, ny, nz, 1);
    auto volumeExtra = volumeFactory.createExtraVolume(nx, ny, nz, 1);
    auto volumePrimitive = volumeFactory.createPrimitiveVolume(nx, ny, nz, 1);

    // Fill every variable with 42
    const std::string pythonCode =
        "def init_global(rho, ux, uy, uz, p, nx, ny, nz, ax, ay, az, bx, by, bz):\n"
        "    for k in range(nz):\n"
        "        for j in range(ny):\n"
        "            for i in range(nx):\n"
        "                rho[i,j,k] = k*nx*ny + j*nx + i\n"
        "                ux[i,j,k] = 2*(k*nx*ny + j*nx + i)\n"
        "                uy[i,j,k] = 3*(k*nx*ny + j*nx + i)\n"
        "                uz[i,j,k] = 4*(k*nx*ny + j*nx + i)\n"
        "                p[i,j,k] = 5*(k*nx*ny + j*nx + i)\n";

    PythonInitialData initialData(pythonCode, Parameters());

    initialData.setInitialData(*volumeConserved,
        *volumeExtra,
        *volumePrimitive,
        *cellComputer,
        grid);


    const auto ngx = volumePrimitive->getNumberOfXGhostCells();
    const auto ngy = volumePrimitive->getNumberOfYGhostCells();
    const auto ngz = volumePrimitive->getNumberOfZGhostCells();

    for (size_t var = 0; var < volumePrimitive->getNumberOfVariables(); ++var) {
        for (size_t k = 0; k < nz; ++k) {
            for (size_t j = 0; j < ny; ++j) {
                for (size_t i = 0; i < nx; ++i) {
                    const auto innerIndex = k * nx * ny + j * nx + i;
                    const auto outerIndex =
                        (k + ngz) * (nx + 2 * ngx) * (ny + 2 * ngy)
                        + (j + ngy) * (nx + 2 * ngx)
                        + (i + ngx);
                    ASSERT_EQ((var + 1) * innerIndex,
                        volumePrimitive->getScalarMemoryArea(var)->getPointer()[outerIndex]);
                }
            }

        }
    }
}



TEST(PythonInitialDataTest, Index2D) {


    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration(
        new DeviceConfiguration);
    auto simulatorParameters =
        alsfvm::make_shared<simulator::SimulatorParameters>("euler3", "cpu");
    equation::CellComputerFactory cellComputerFactory(simulatorParameters,
        deviceConfiguration);
    auto cellComputer = cellComputerFactory.createComputer();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);
    volume::VolumeFactory volumeFactory("euler3", memoryFactory);
    size_t nx = 10;
    size_t ny = 10;
    size_t nz = 1;

    grid::Grid grid({0., 0., 0.}, {1, 1, 1}, {int(nx), int(ny), int(nz)});


    auto volumeConserved = volumeFactory.createConservedVolume(nx, ny, nz, 1);
    auto volumeExtra = volumeFactory.createExtraVolume(nx, ny, nz, 1);
    auto volumePrimitive = volumeFactory.createPrimitiveVolume(nx, ny, nz, 1);

    // Fill every variable with 42
    const std::string pythonCode =
        "def init_global(rho, ux, uy, uz, p, nx, ny, nz, ax, ay, az, bx, by, bz):\n"
        "    for k in range(nz):\n"
        "        for j in range(ny):\n"
        "            for i in range(nx):\n"
        "                rho[i,j,k] = k*nx*ny + j*nx + i\n"
        "                ux[i,j,k] = 2*(k*nx*ny + j*nx + i)\n"
        "                uy[i,j,k] = 3*(k*nx*ny + j*nx + i)\n"
        "                uz[i,j,k] = 4*(k*nx*ny + j*nx + i)\n"
        "                p[i,j,k] = 5*(k*nx*ny + j*nx + i)\n";

    PythonInitialData initialData(pythonCode, Parameters());

    initialData.setInitialData(*volumeConserved,
        *volumeExtra,
        *volumePrimitive,
        *cellComputer,
        grid);


    const auto ngx = volumePrimitive->getNumberOfXGhostCells();
    const auto ngy = volumePrimitive->getNumberOfYGhostCells();
    const auto ngz = volumePrimitive->getNumberOfZGhostCells();

    for (size_t var = 0; var < volumePrimitive->getNumberOfVariables(); ++var) {
        for (size_t k = 0; k < nz; ++k) {
            for (size_t j = 0; j < ny; ++j) {
                for (size_t i = 0; i < nx; ++i) {
                    const auto innerIndex = k * nx * ny + j * nx + i;
                    const auto outerIndex =
                        (k + ngz) * (nx + 2 * ngx) * (ny + 2 * ngy)
                        + (j + ngy) * (nx + 2 * ngx)
                        + (i + ngx);
                    ASSERT_EQ((var + 1) * innerIndex,
                        volumePrimitive->getScalarMemoryArea(var)->getPointer()[outerIndex]);
                }
            }

        }
    }
}



TEST(PythonInitialDataTest, Index1D) {


    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration(
        new DeviceConfiguration);
    auto simulatorParameters =
        alsfvm::make_shared<simulator::SimulatorParameters>("euler3", "cpu");
    equation::CellComputerFactory cellComputerFactory(simulatorParameters,
        deviceConfiguration);
    auto cellComputer = cellComputerFactory.createComputer();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);
    volume::VolumeFactory volumeFactory("euler3", memoryFactory);
    size_t nx = 10;
    size_t ny = 1;
    size_t nz = 1;

    grid::Grid grid({0., 0., 0.}, {1, 1, 1}, {int(nx), int(ny), int(nz)});


    auto volumeConserved = volumeFactory.createConservedVolume(nx, ny, nz, 1);
    auto volumeExtra = volumeFactory.createExtraVolume(nx, ny, nz, 1);
    auto volumePrimitive = volumeFactory.createPrimitiveVolume(nx, ny, nz, 1);

    // Fill every variable with 42
    const std::string pythonCode =
        "def init_global(rho, ux, uy, uz, p, nx, ny, nz, ax, ay, az, bx, by, bz):\n"
        "    for k in range(nz):\n"
        "        for j in range(ny):\n"
        "            for i in range(nx):\n"
        "                rho[i,j,k] = k*nx*ny + j*nx + i\n"
        "                ux[i,j,k] = 2*(k*nx*ny + j*nx + i)\n"
        "                uy[i,j,k] = 3*(k*nx*ny + j*nx + i)\n"
        "                uz[i,j,k] = 4*(k*nx*ny + j*nx + i)\n"
        "                p[i,j,k] = 5*(k*nx*ny + j*nx + i)\n";

    PythonInitialData initialData(pythonCode, Parameters());

    initialData.setInitialData(*volumeConserved,
        *volumeExtra,
        *volumePrimitive,
        *cellComputer,
        grid);


    const auto ngx = volumePrimitive->getNumberOfXGhostCells();
    const auto ngy = volumePrimitive->getNumberOfYGhostCells();
    const auto ngz = volumePrimitive->getNumberOfZGhostCells();

    for (size_t var = 0; var < volumePrimitive->getNumberOfVariables(); ++var) {
        for (size_t k = 0; k < nz; ++k) {
            for (size_t j = 0; j < ny; ++j) {
                for (size_t i = 0; i < nx; ++i) {
                    const auto innerIndex = k * nx * ny + j * nx + i;
                    const auto outerIndex =
                        (k + ngz) * (nx + 2 * ngx) * (ny + 2 * ngy)
                        + (j + ngy) * (nx + 2 * ngx)
                        + (i + ngx);
                    ASSERT_EQ((var + 1) * innerIndex,
                        volumePrimitive->getScalarMemoryArea(var)->getPointer()[outerIndex]);
                }
            }

        }
    }
}

