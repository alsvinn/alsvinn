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


    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration(new DeviceConfiguration);
    auto simulatorParameters = alsfvm::make_shared<simulator::SimulatorParameters>("euler3", "cpu");
    equation::CellComputerFactory cellComputerFactory(simulatorParameters, deviceConfiguration);
    auto cellComputer = cellComputerFactory.createComputer();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);
    volume::VolumeFactory volumeFactory("euler3", memoryFactory);
    size_t nx = 10;
    size_t ny = 10;
    size_t nz = 1;

    grid::Grid grid({0.,0.,0.}, {1,1,1}, {int(nx), int(ny), int(nz)});


    auto volumeConserved = volumeFactory.createConservedVolume(nx, ny, nz, 1);
    auto volumeExtra = volumeFactory.createExtraVolume(nx, ny, nz, 1);
    auto volumePrimitive = volumeFactory.createPrimitiveVolume(nx, ny, nz, 1);

    // Fill every variable with 42
    const std::string pythonCode = "rho = 42\nux=42\nuy=42\nuz=42\np=42";

    PythonInitialData initialData(pythonCode);

    initialData.setInitialData(*volumeConserved,
                               *volumeExtra,
                               *volumePrimitive,
                               *cellComputer,
                               grid);

    volume::for_each_midpoint(*volumePrimitive, grid, [&](real x, real y, real z, size_t index){
        for(size_t var = 0; var < volumePrimitive->getNumberOfVariables(); ++var) {
            ASSERT_EQ(42, volumePrimitive->getScalarMemoryArea(var)->getPointer()[index]);
        }
    });

}

TEST(PythonInitialDataTest, RiemannProblem) {

    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration(new DeviceConfiguration);
    auto simulatorParameters = alsfvm::make_shared<simulator::SimulatorParameters>("euler3", "cpu");
    equation::CellComputerFactory cellComputerFactory(simulatorParameters, deviceConfiguration);
    auto cellComputer = cellComputerFactory.createComputer();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);
    volume::VolumeFactory volumeFactory("euler3", memoryFactory);
    size_t nx = 10;
    size_t ny = 10;
    size_t nz = 1;

    grid::Grid grid({0.,0.,0.}, {1,1,1}, {int(nx), int(ny), int(nz)});


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


    PythonInitialData initialData(pythonCode);

    initialData.setInitialData(*volumeConserved,
                               *volumeExtra,
                               *volumePrimitive,
                               *cellComputer,
                               grid);

    volume::for_each_midpoint(*volumePrimitive, grid, [&](real x, real y, real z, size_t index){
        for(size_t var = 0; var < volumePrimitive->getNumberOfVariables(); ++var) {
            if (x > 0.5) {
                ASSERT_EQ(42, volumePrimitive->getScalarMemoryArea(var)->getPointer()[index]);
            } else {
                ASSERT_EQ(2, volumePrimitive->getScalarMemoryArea(var)->getPointer()[index]);
            }
        }
    });

}


TEST(PythonInitialDataTest, SineCosineExp) {

    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration(new DeviceConfiguration);
    auto simulatorParameters = alsfvm::make_shared<simulator::SimulatorParameters>("euler3", "cpu");
    equation::CellComputerFactory cellComputerFactory(simulatorParameters, deviceConfiguration);
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
        

    PythonInitialData initialData(pythonCode);

    initialData.setInitialData(*volumeConserved,
        *volumeExtra,
        *volumePrimitive,
        *cellComputer,
        grid);

    volume::for_each_midpoint(*volumePrimitive, grid, [&](real x, real y, real z, size_t index){
        ASSERT_EQ(std::sin(x), volumePrimitive->getScalarMemoryArea("rho")->getPointer()[index]);
        ASSERT_EQ(std::cos(x), volumePrimitive->getScalarMemoryArea("ux")->getPointer()[index]);
        ASSERT_EQ(std::exp(x), volumePrimitive->getScalarMemoryArea("uy")->getPointer()[index]);
        ASSERT_EQ(x, volumePrimitive->getScalarMemoryArea("uz")->getPointer()[index]);
        ASSERT_EQ(y, volumePrimitive->getScalarMemoryArea("p")->getPointer()[index]);
        
    });

}
