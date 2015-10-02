#include <gtest/gtest.h>

#include "alsfvm/reconstruction/WENOCUDA.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/reconstruction/WENOCoefficients.hpp"

using namespace alsfvm;
using namespace alsfvm::memory;
using namespace alsfvm::volume;
using namespace alsfvm::reconstruction;
using namespace alsfvm::grid;

TEST(CUDAWenoTest, ConstantZeroTestSecondOrder) {
	const size_t nx = 10, ny = 10, nz = 1;

	auto deviceConfiguration = std::make_shared<DeviceConfiguration>("cuda");
	auto memoryFactory = std::make_shared<MemoryFactory>(deviceConfiguration);

	VolumeFactory volumeFactory("euler", memoryFactory);
	WENOCUDA<equation::euler::Euler, 2> wenoCUDA;

	auto conserved = volumeFactory.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());
	auto left = volumeFactory.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());
	auto right = volumeFactory.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());

	conserved->makeZero();



	wenoCUDA.performReconstruction(*conserved, 0, 0, *left, *right);


	auto deviceConfigurationCPU = std::make_shared<DeviceConfiguration>("cpu");
	auto memoryFactoryCPU = std::make_shared<MemoryFactory>(deviceConfigurationCPU);

	VolumeFactory volumeFactoryCPU("euler", memoryFactoryCPU);
	auto rightCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());
	auto leftCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());

	right->copyTo(*rightCPU);
	left->copyTo(*leftCPU);
	for_each_internal_volume_index(*left, 0, [&](size_t, size_t middle, size_t) {
		ASSERT_EQ(0, leftCPU->getScalarMemoryArea(0)->getPointer()[middle]);
		ASSERT_EQ(0, leftCPU->getScalarMemoryArea(1)->getPointer()[middle]);
		ASSERT_EQ(0, leftCPU->getScalarMemoryArea(2)->getPointer()[middle]);
		ASSERT_EQ(0, leftCPU->getScalarMemoryArea(3)->getPointer()[middle]);
		ASSERT_EQ(0, leftCPU->getScalarMemoryArea(4)->getPointer()[middle]);

		ASSERT_EQ(0, rightCPU->getScalarMemoryArea(0)->getPointer()[middle]);
		ASSERT_EQ(0, rightCPU->getScalarMemoryArea(1)->getPointer()[middle]);
		ASSERT_EQ(0, rightCPU->getScalarMemoryArea(2)->getPointer()[middle]);
		ASSERT_EQ(0, rightCPU->getScalarMemoryArea(3)->getPointer()[middle]);
		ASSERT_EQ(0, rightCPU->getScalarMemoryArea(4)->getPointer()[middle]);
	});
}

TEST(CUDAWenoTest, ConstantZeroTestThirdOrder) {
	const size_t nx = 10, ny = 10, nz = 1;

	auto deviceConfiguration = std::make_shared<DeviceConfiguration>("cuda");
	auto memoryFactory = std::make_shared<MemoryFactory>(deviceConfiguration);

	VolumeFactory volumeFactory("euler", memoryFactory);
	WENOCUDA<equation::euler::Euler, 3> wenoCUDA;

	auto conserved = volumeFactory.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());
	auto left = volumeFactory.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());
	auto right = volumeFactory.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());

	conserved->makeZero();



	wenoCUDA.performReconstruction(*conserved, 0, 0, *left, *right);

	auto deviceConfigurationCPU = std::make_shared<DeviceConfiguration>("cpu");
	auto memoryFactoryCPU = std::make_shared<MemoryFactory>(deviceConfigurationCPU);

	VolumeFactory volumeFactoryCPU("euler", memoryFactoryCPU);
	auto rightCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());
	auto leftCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());

	right->copyTo(*rightCPU);
	left->copyTo(*leftCPU);

	for_each_internal_volume_index(*left, 0, [&](size_t, size_t middle, size_t) {
		ASSERT_EQ(0, leftCPU->getScalarMemoryArea(0)->getPointer()[middle]);
		ASSERT_EQ(0, leftCPU->getScalarMemoryArea(1)->getPointer()[middle]);
		ASSERT_EQ(0, leftCPU->getScalarMemoryArea(2)->getPointer()[middle]);
		ASSERT_EQ(0, leftCPU->getScalarMemoryArea(3)->getPointer()[middle]);
		ASSERT_EQ(0, leftCPU->getScalarMemoryArea(4)->getPointer()[middle]);

		ASSERT_EQ(0, rightCPU->getScalarMemoryArea(0)->getPointer()[middle]);
		ASSERT_EQ(0, rightCPU->getScalarMemoryArea(1)->getPointer()[middle]);
		ASSERT_EQ(0, rightCPU->getScalarMemoryArea(2)->getPointer()[middle]);
		ASSERT_EQ(0, rightCPU->getScalarMemoryArea(3)->getPointer()[middle]);
		ASSERT_EQ(0, rightCPU->getScalarMemoryArea(4)->getPointer()[middle]);
	});
}

TEST(CUDAWenoTest, ConstantOneTestSecondOrder) {
	WENOCUDA<equation::euler::Euler, 2> wenoCUDA;
	const size_t nx = 10, ny = 10, nz = 1;
	auto deviceConfigurationCPU = std::make_shared<DeviceConfiguration>("cpu");
	auto memoryFactoryCPU = std::make_shared<MemoryFactory>(deviceConfigurationCPU);

	VolumeFactory volumeFactoryCPU("euler", memoryFactoryCPU);
	auto rightCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());
	auto leftCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());
	auto conservedCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());

	auto deviceConfiguration = std::make_shared<DeviceConfiguration>("cuda");
	auto memoryFactory = std::make_shared<MemoryFactory>(deviceConfiguration);

	VolumeFactory volumeFactory("euler", memoryFactory);

	auto conserved = volumeFactory.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());
	auto left = volumeFactory.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());
	auto right = volumeFactory.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());

	for_each_cell_index(*conservedCPU, [&](size_t index) {
		conservedCPU->getScalarMemoryArea("rho")->getPointer()[index] = 1;
		conservedCPU->getScalarMemoryArea("mx")->getPointer()[index] = 1;
		conservedCPU->getScalarMemoryArea("my")->getPointer()[index] = 1;
		conservedCPU->getScalarMemoryArea("mz")->getPointer()[index] = 1;
		conservedCPU->getScalarMemoryArea("E")->getPointer()[index] = 10;

	});

	conservedCPU->copyTo(*conserved);


	wenoCUDA.performReconstruction(*conserved, 0, 0, *left, *right);

	left->copyTo(*leftCPU);
	right->copyTo(*rightCPU);

	for_each_internal_volume_index(*left, 0, [&](size_t, size_t middle, size_t) {
		ASSERT_NEAR(1,  leftCPU->getScalarMemoryArea(0)->getPointer()[middle], 1e-8);
		ASSERT_NEAR(1,  leftCPU->getScalarMemoryArea(1)->getPointer()[middle], 1e-8);
		ASSERT_NEAR(1,  leftCPU->getScalarMemoryArea(2)->getPointer()[middle], 1e-8);
		ASSERT_NEAR(1,  leftCPU->getScalarMemoryArea(3)->getPointer()[middle], 1e-8);
		ASSERT_NEAR(10, leftCPU->getScalarMemoryArea(4)->getPointer()[middle], 1e-8);

		ASSERT_NEAR(1,  rightCPU->getScalarMemoryArea(0)->getPointer()[middle], 1e-8);
		ASSERT_NEAR(1,  rightCPU->getScalarMemoryArea(1)->getPointer()[middle], 1e-8);
		ASSERT_NEAR(1,  rightCPU->getScalarMemoryArea(2)->getPointer()[middle], 1e-8);
		ASSERT_NEAR(1,  rightCPU->getScalarMemoryArea(3)->getPointer()[middle], 1e-8);
		ASSERT_NEAR(10, rightCPU->getScalarMemoryArea(4)->getPointer()[middle], 1e-8);
	});
}

TEST(CUDAWenoTest, ConstantOneTestThirdOrder) {
	WENOCUDA<equation::euler::Euler, 3> wenoCUDA;
	const size_t nx = 10, ny = 10, nz = 1;
	auto deviceConfigurationCPU = std::make_shared<DeviceConfiguration>("cpu");
	auto memoryFactoryCPU = std::make_shared<MemoryFactory>(deviceConfigurationCPU);

	VolumeFactory volumeFactoryCPU("euler", memoryFactoryCPU);
	auto rightCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());
	auto leftCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());
	auto conservedCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());

	auto deviceConfiguration = std::make_shared<DeviceConfiguration>("cuda");
	auto memoryFactory = std::make_shared<MemoryFactory>(deviceConfiguration);

	VolumeFactory volumeFactory("euler", memoryFactory);

	auto conserved = volumeFactory.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());
	auto left = volumeFactory.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());
	auto right = volumeFactory.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());

	for_each_cell_index(*conservedCPU, [&](size_t index) {
		conservedCPU->getScalarMemoryArea("rho")->getPointer()[index] = 1;
		conservedCPU->getScalarMemoryArea("mx")->getPointer()[index] = 1;
		conservedCPU->getScalarMemoryArea("my")->getPointer()[index] = 1;
		conservedCPU->getScalarMemoryArea("mz")->getPointer()[index] = 1;
		conservedCPU->getScalarMemoryArea("E")->getPointer()[index] = 10;

	});

	conservedCPU->copyTo(*conserved);


	wenoCUDA.performReconstruction(*conserved, 0, 0, *left, *right);

	left->copyTo(*leftCPU);
	right->copyTo(*rightCPU);

	for_each_internal_volume_index(*left, 0, [&](size_t, size_t middle, size_t) {
		ASSERT_NEAR(1, leftCPU->getScalarMemoryArea(0)->getPointer()[middle], 1e-8);
		ASSERT_NEAR(1, leftCPU->getScalarMemoryArea(1)->getPointer()[middle], 1e-8);
		ASSERT_NEAR(1, leftCPU->getScalarMemoryArea(2)->getPointer()[middle], 1e-8);
		ASSERT_NEAR(1, leftCPU->getScalarMemoryArea(3)->getPointer()[middle], 1e-8);
		ASSERT_NEAR(10, leftCPU->getScalarMemoryArea(4)->getPointer()[middle], 1e-8);

		ASSERT_NEAR(1, rightCPU->getScalarMemoryArea(0)->getPointer()[middle], 1e-8);
		ASSERT_NEAR(1, rightCPU->getScalarMemoryArea(1)->getPointer()[middle], 1e-8);
		ASSERT_NEAR(1, rightCPU->getScalarMemoryArea(2)->getPointer()[middle], 1e-8);
		ASSERT_NEAR(1, rightCPU->getScalarMemoryArea(3)->getPointer()[middle], 1e-8);
		ASSERT_NEAR(10, rightCPU->getScalarMemoryArea(4)->getPointer()[middle], 1e-8);
	});
}

TEST(CUDAWenoTest, ReconstructionSimple) {
	const size_t nx = 10, ny = 1, nz = 1;

	auto deviceConfiguration = std::make_shared<DeviceConfiguration>();
	auto memoryFactory = std::make_shared<MemoryFactory>(deviceConfiguration);

	VolumeFactory volumeFactory("euler", memoryFactory);
	WENOCUDA<equation::euler::Euler, 2> wenoCUDA;
	auto conserved = volumeFactory.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());
	auto left = volumeFactory.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());
	auto right = volumeFactory.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());

	auto deviceConfigurationCPU = std::make_shared<DeviceConfiguration>("cpu");
	auto memoryFactoryCPU = std::make_shared<MemoryFactory>(deviceConfigurationCPU);

	VolumeFactory volumeFactoryCPU("euler", memoryFactoryCPU);
	auto rightCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());
	auto leftCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());
	auto conservedCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, wenoCUDA.getNumberOfGhostCells());

	for_each_cell_index(*conservedCPU, [&](size_t index) {
		// fill some dummy data
		conservedCPU->getScalarMemoryArea("rho")->getPointer()[index] = 1;
		conservedCPU->getScalarMemoryArea("mx")->getPointer()[index] = 1;
		conservedCPU->getScalarMemoryArea("my")->getPointer()[index] = 1;
		conservedCPU->getScalarMemoryArea("mz")->getPointer()[index] = 1;
		conservedCPU->getScalarMemoryArea("E")->getPointer()[index] = 10;

	});


	// This is the main ingredient:
	conservedCPU->getScalarMemoryArea("rho")->getPointer()[1] = 2;
	conservedCPU->getScalarMemoryArea("rho")->getPointer()[2] = 0;
	conservedCPU->getScalarMemoryArea("rho")->getPointer()[3] = 1;

	conservedCPU->copyTo(*conserved);


	wenoCUDA.performReconstruction(*conserved, 0, 0, *left, *right);

	const real epsilon = WENOCoefficients<2>::epsilon;
	const real right1 = 0.5;
	const real right2 = -1;

	const real left1 = -0.5;
	const real left2 = 1;

	const real d0 = 2.0 / 3.0;
	const real d1 = 1.0 / 3.0;

	const real beta0 = 4.0;
	const real beta1 = 1.0;

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

	right->copyTo(*rightCPU);
	left->copyTo(*leftCPU);

	ASSERT_NEAR(omega0 * right1 + omega1 * right2, rightCPU->getScalarMemoryArea("rho")->getPointer()[2], 1e-8);
	ASSERT_NEAR(omega0Tilde * left1 + omega1Tilde * left2, leftCPU->getScalarMemoryArea("rho")->getPointer()[2], 1e-8);

}
