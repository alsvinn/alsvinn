#include <gtest/gtest.h>

#include "alsfvm/reconstruction/WENOCPU.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/reconstruction/WENOCoefficients.hpp"
#include "alsfvm/reconstruction/ReconstructionFactory.hpp"

using namespace alsfvm;
using namespace alsfvm::memory;
using namespace alsfvm::volume;
using namespace alsfvm::reconstruction;
using namespace alsfvm::grid;

TEST(WenoTest, ConstantZeroTestSecondOrder) {
    const size_t nx = 10, ny = 10, nz = 1;

    auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("euler3", memoryFactory);
    ReconstructionFactory reconstructionFactory;
    grid::Grid grid(rvec3(0, 0, 0), rvec3(1, 1, 1), ivec3(nx, ny, nz));
    simulator::SimulatorParameters parameters;
    parameters.setEquationParameters(
        alsfvm::dynamic_pointer_cast<equation::EquationParameters>(
            alsfvm::make_shared<equation::euler::EulerParameters>()));
    auto wenoCPU = reconstructionFactory.createReconstruction("weno2", "euler3",
            parameters, memoryFactory,  grid, deviceConfiguration);


    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz,
            wenoCPU->getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz,
            wenoCPU->getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz,
            wenoCPU->getNumberOfGhostCells());

    conserved->makeZero();



    wenoCPU->performReconstruction(*conserved, 0, 0, *left, *right);
    const int ngx =  wenoCPU->getNumberOfGhostCells();
    const int ngy =  wenoCPU->getNumberOfGhostCells();
    const int ngz = 0;
    for_each_cell_index(*left,  [&](size_t middle ) {
        ASSERT_EQ(0, left->getScalarMemoryArea(0)->getPointer()[middle]);
        ASSERT_EQ(0, left->getScalarMemoryArea(1)->getPointer()[middle]);
        ASSERT_EQ(0, left->getScalarMemoryArea(2)->getPointer()[middle]);
        ASSERT_EQ(0, left->getScalarMemoryArea(3)->getPointer()[middle]);
        ASSERT_EQ(0, left->getScalarMemoryArea(4)->getPointer()[middle]);

        ASSERT_EQ(0, right->getScalarMemoryArea(0)->getPointer()[middle]);
        ASSERT_EQ(0, right->getScalarMemoryArea(1)->getPointer()[middle]);
        ASSERT_EQ(0, right->getScalarMemoryArea(2)->getPointer()[middle]);
        ASSERT_EQ(0, right->getScalarMemoryArea(3)->getPointer()[middle]);
        ASSERT_EQ(0, right->getScalarMemoryArea(4)->getPointer()[middle]);
    }, {ngx - 1, ngy, ngz}, {ngx - 1, ngy, ngz});
}

TEST(WenoTest, ConstantZeroTestThirdOrder) {
    const size_t nx = 10, ny = 10, nz = 1;

    auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("euler3", memoryFactory);
    ReconstructionFactory reconstructionFactory;
    grid::Grid grid(rvec3(0, 0, 0), rvec3(1, 1, 1), ivec3(nx, ny, nz));
    simulator::SimulatorParameters parameters;
    parameters.setEquationParameters(
        alsfvm::dynamic_pointer_cast<equation::EquationParameters>(
            alsfvm::make_shared<equation::euler::EulerParameters>()));
    auto wenoCPU = reconstructionFactory.createReconstruction("weno2", "euler3",
            parameters, memoryFactory,  grid, deviceConfiguration);
    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz,
            wenoCPU->getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz,
            wenoCPU->getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz,
            wenoCPU->getNumberOfGhostCells());

    conserved->makeZero();



    wenoCPU->performReconstruction(*conserved, 0, 0, *left, *right);
    const int ngx =  wenoCPU->getNumberOfGhostCells();
    const int ngy =  wenoCPU->getNumberOfGhostCells();
    const int ngz = 0;
    for_each_cell_index(*left, [&](size_t middle ) {
        ASSERT_EQ(0, left->getScalarMemoryArea(0)->getPointer()[middle]);
        ASSERT_EQ(0, left->getScalarMemoryArea(1)->getPointer()[middle]);
        ASSERT_EQ(0, left->getScalarMemoryArea(2)->getPointer()[middle]);
        ASSERT_EQ(0, left->getScalarMemoryArea(3)->getPointer()[middle]);
        ASSERT_EQ(0, left->getScalarMemoryArea(4)->getPointer()[middle]);

        ASSERT_EQ(0, right->getScalarMemoryArea(0)->getPointer()[middle]);
        ASSERT_EQ(0, right->getScalarMemoryArea(1)->getPointer()[middle]);
        ASSERT_EQ(0, right->getScalarMemoryArea(2)->getPointer()[middle]);
        ASSERT_EQ(0, right->getScalarMemoryArea(3)->getPointer()[middle]);
        ASSERT_EQ(0, right->getScalarMemoryArea(4)->getPointer()[middle]);
    }, {ngx - 1, ngy, ngz}, {ngx - 1, ngy, ngz});
}

TEST(WenoTest, ConstantOneTestSecondOrder) {

    const size_t nx = 10, ny = 10, nz = 1;

    auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("euler3", memoryFactory);
    ReconstructionFactory reconstructionFactory;
    grid::Grid grid(rvec3(0, 0, 0), rvec3(1, 1, 1), ivec3(nx, ny, nz));
    simulator::SimulatorParameters parameters;
    parameters.setEquationParameters(
        alsfvm::dynamic_pointer_cast<equation::EquationParameters>(
            alsfvm::make_shared<equation::euler::EulerParameters>()));
    auto wenoCPU = reconstructionFactory.createReconstruction("weno2", "euler3",
            parameters, memoryFactory,  grid, deviceConfiguration);
    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz,
            wenoCPU->getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz,
            wenoCPU->getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz,
            wenoCPU->getNumberOfGhostCells());

    for_each_cell_index(*conserved, [&] (size_t index) {
        conserved->getScalarMemoryArea("rho")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mx")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("my")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mz")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("E")->getPointer()[index] = 10;

    });


    wenoCPU->performReconstruction(*conserved, 0, 0, *left, *right);
    const int ngx =  wenoCPU->getNumberOfGhostCells();
    const int ngy =  wenoCPU->getNumberOfGhostCells();
    const int ngz = 0;
    for_each_cell_index(*left, [&](size_t middle ) {
        ASSERT_NEAR(1, left->getScalarMemoryArea(0)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1, left->getScalarMemoryArea(1)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1, left->getScalarMemoryArea(2)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1, left->getScalarMemoryArea(3)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(10, left->getScalarMemoryArea(4)->getPointer()[middle], 1e-8);

        ASSERT_NEAR(1, right->getScalarMemoryArea(0)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1, right->getScalarMemoryArea(1)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1, right->getScalarMemoryArea(2)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1, right->getScalarMemoryArea(3)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(10, right->getScalarMemoryArea(4)->getPointer()[middle], 1e-8);
    }, {ngx - 1, ngy, ngz}, {ngx - 1, ngy, ngz});
}

TEST(WenoTest, ConstantOneTestThirdOrder) {

    const size_t nx = 10, ny = 10, nz = 1;

    auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("euler3", memoryFactory);
    ReconstructionFactory reconstructionFactory;
    grid::Grid grid(rvec3(0, 0, 0), rvec3(1, 1, 1), ivec3(nx, ny, nz));
    simulator::SimulatorParameters parameters;
    parameters.setEquationParameters(
        alsfvm::dynamic_pointer_cast<equation::EquationParameters>(
            alsfvm::make_shared<equation::euler::EulerParameters>()));
    auto wenoCPU = reconstructionFactory.createReconstruction("weno2", "euler3",
            parameters, memoryFactory,  grid, deviceConfiguration);
    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz,
            wenoCPU->getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz,
            wenoCPU->getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz,
            wenoCPU->getNumberOfGhostCells());

    for_each_cell_index(*conserved, [&] (size_t index) {
        conserved->getScalarMemoryArea("rho")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mx")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("my")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mz")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("E")->getPointer()[index] = 10;

    });


    wenoCPU->performReconstruction(*conserved, 0, 0, *left, *right);
    const int ngx =  wenoCPU->getNumberOfGhostCells();
    const int ngy =  wenoCPU->getNumberOfGhostCells();
    const int ngz = 0;
    for_each_cell_index(*left, [&](size_t middle ) {
        ASSERT_NEAR(1, left->getScalarMemoryArea(0)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1, left->getScalarMemoryArea(1)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1, left->getScalarMemoryArea(2)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1, left->getScalarMemoryArea(3)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(10, left->getScalarMemoryArea(4)->getPointer()[middle], 1e-8);

        ASSERT_NEAR(1, right->getScalarMemoryArea(0)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1, right->getScalarMemoryArea(1)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1, right->getScalarMemoryArea(2)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1, right->getScalarMemoryArea(3)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(10, right->getScalarMemoryArea(4)->getPointer()[middle], 1e-8);
    }, {ngx - 1, ngy, ngz}, {ngx - 1, ngy, ngz});
}

TEST(WenoTest, ReconstructionSimple) {
    const size_t nx = 10, ny = 1, nz = 1;

    auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("euler3", memoryFactory);
    ReconstructionFactory reconstructionFactory;
    grid::Grid grid(rvec3(0, 0, 0), rvec3(1, 1, 1), ivec3(nx, ny, nz));
    simulator::SimulatorParameters parameters;
    parameters.setEquationParameters(
        alsfvm::dynamic_pointer_cast<equation::EquationParameters>(
            alsfvm::make_shared<equation::euler::EulerParameters>()));
    auto wenoCPU = reconstructionFactory.createReconstruction("weno2", "euler3",
            parameters, memoryFactory,  grid, deviceConfiguration);
    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz,
            wenoCPU->getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz,
            wenoCPU->getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz,
            wenoCPU->getNumberOfGhostCells());
    for_each_cell_index(*conserved, [&] (size_t index) {
        // fill some dummy data
        conserved->getScalarMemoryArea("rho")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mx")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("my")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mz")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("E")->getPointer()[index] = 10;

    });

    // This is the main ingredient:
    conserved->getScalarMemoryArea("rho")->getPointer()[1] = 2;
    conserved->getScalarMemoryArea("rho")->getPointer()[2] = 0;
    conserved->getScalarMemoryArea("rho")->getPointer()[3] = 1;

    wenoCPU->performReconstruction(*conserved, 0, 0, *left, *right);

    const real epsilon = WENOCoefficients<2>::epsilon;
    const real right1 = 0.5;
    const real right2 = -1;

    const real left1 = -0.5;
    const real left2 = 1;

    const real d0 = 2.0 / 3.0;
    const real d1 = 1.0 / 3.0;

    const real beta0 = 1.0;
    const real beta1 = 4.0;

    const real alpha0 = d0 / pow(beta0 + epsilon, 2);
    const real alpha1 = d1 / pow(beta1 + epsilon, 2);
    const real alphaSum = alpha0 + alpha1;

    const real alpha0Tilde = d1 / pow(beta0 + epsilon, 2);
    const real alpha1Tilde = d0 / pow(beta1 + epsilon, 2);
    const real alphaTildeSum = alpha0Tilde + alpha1Tilde;


    const real omega0 = alpha0 / alphaSum;
    const real omega1 = alpha1 / alphaSum;

    const real omega0Tilde = alpha0Tilde / alphaTildeSum;
    const real omega1Tilde = alpha1Tilde / alphaTildeSum;

    ASSERT_NEAR(omega0 * right1 + omega1 * right2,
        right->getScalarMemoryArea("rho")->getPointer()[2], 1e-8);
    ASSERT_NEAR(omega0Tilde * left1 + omega1Tilde * left2,
        left->getScalarMemoryArea("rho")->getPointer()[2], 1e-8);

}


TEST(WenoTest, ReconstructionSimpleYDirection) {
    const size_t nx = 10, ny = 10, nz = 1;

    auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("euler3", memoryFactory);
    ReconstructionFactory reconstructionFactory;
    grid::Grid grid(rvec3(0, 0, 0), rvec3(1, 1, 1), ivec3(nx, ny, nz));
    simulator::SimulatorParameters parameters;
    parameters.setEquationParameters(
        alsfvm::dynamic_pointer_cast<equation::EquationParameters>(
            alsfvm::make_shared<equation::euler::EulerParameters>()));
    auto wenoCPU = reconstructionFactory.createReconstruction("weno2", "euler3",
            parameters, memoryFactory,  grid, deviceConfiguration);
    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz,
            wenoCPU->getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz,
            wenoCPU->getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz,
            wenoCPU->getNumberOfGhostCells());
    for_each_cell_index(*conserved, [&] (size_t index) {
        // fill some dummy data
        conserved->getScalarMemoryArea("rho")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mx")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("my")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mz")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("E")->getPointer()[index] = 10;

    });

    const size_t totalNx = nx + 2 * wenoCPU->getNumberOfGhostCells();
    // This is the main ingredient:
    conserved->getScalarMemoryArea("rho")->getPointer()[3 + 1 * totalNx] = 2;
    conserved->getScalarMemoryArea("rho")->getPointer()[3 + 2 * totalNx] = 0;
    conserved->getScalarMemoryArea("rho")->getPointer()[3 + 3 * totalNx] = 1;

    wenoCPU->performReconstruction(*conserved, 1, 0, *left, *right);

    const real epsilon = WENOCoefficients<2>::epsilon;
    const real right1 = 0.5;
    const real right2 = -1;

    const real left1 = -0.5;
    const real left2 = 1;

    const real d0 = 2.0 / 3.0;
    const real d1 = 1.0 / 3.0;

    const real beta0 = 1.0;
    const real beta1 = 4.0;

    const real alpha0 = d0 / pow(beta0 + epsilon, 2);
    const real alpha1 = d1 / pow(beta1 + epsilon, 2);
    const real alphaSum = alpha0 + alpha1;

    const real alpha0Tilde = d1 / pow(beta0 + epsilon, 2);
    const real alpha1Tilde = d0 / pow(beta1 + epsilon, 2);
    const real alphaTildeSum = alpha0Tilde + alpha1Tilde;


    const real omega0 = alpha0 / alphaSum;
    const real omega1 = alpha1 / alphaSum;

    const real omega0Tilde = alpha0Tilde / alphaTildeSum;
    const real omega1Tilde = alpha1Tilde / alphaTildeSum;

    ASSERT_NEAR(omega0 * right1 + omega1 * right2,
        right->getScalarMemoryArea("rho")->getPointer()[3 + 2 * totalNx], 1e-8);
    ASSERT_NEAR(omega0Tilde * left1 + omega1Tilde * left2,
        left->getScalarMemoryArea("rho")->getPointer()[3 + 2 * totalNx], 1e-8);

}

