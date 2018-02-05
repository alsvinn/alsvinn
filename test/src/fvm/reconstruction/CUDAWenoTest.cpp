#include <gtest/gtest.h>
#include "alsfvm/reconstruction/ReconstructionFactory.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/reconstruction/WENOCoefficients.hpp"
#include "alsfvm/equation/euler/EulerParameters.hpp"

using namespace alsfvm;
using namespace alsfvm::memory;
using namespace alsfvm::volume;
using namespace alsfvm::reconstruction;
using namespace alsfvm::grid;

class CUDAWenoTest : public ::testing::Test {
public:
    size_t nx = 10;
    size_t ny = 10;
    size_t nz = 1;
    int ngx = 0;
    int ngy = 0;
    int ngz = 0;

    Grid grid;

    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration;
    alsfvm::shared_ptr<MemoryFactory> memoryFactory;
    ReconstructionFactory reconstructionFactory;
    VolumeFactory volumeFactory;

    alsfvm::shared_ptr<DeviceConfiguration> deviceConfigurationCPU;
    alsfvm::shared_ptr<MemoryFactory> memoryFactoryCPU;
    VolumeFactory volumeFactoryCPU;

    simulator::SimulatorParameters simulatorParameters;

    alsfvm::shared_ptr<Reconstruction> wenoCUDA;

    alsfvm::shared_ptr<Volume> conserved;
    alsfvm::shared_ptr<Volume> left;
    alsfvm::shared_ptr<Volume> right;

    alsfvm::shared_ptr<Volume> conservedCPU;
    alsfvm::shared_ptr<Volume> leftCPU; 
    alsfvm::shared_ptr<Volume> rightCPU;

    

    CUDAWenoTest()
        : grid({ 0, 0, 0 }, { 1, 1, 1 }, ivec3( nx, ny, nz )),
        deviceConfiguration(new DeviceConfiguration("cuda")),
        memoryFactory(new MemoryFactory(deviceConfiguration)),
        volumeFactory("euler3", memoryFactory),
        deviceConfigurationCPU(new DeviceConfiguration("cpu")),
        memoryFactoryCPU(new MemoryFactory(deviceConfigurationCPU)),
        volumeFactoryCPU("euler3", memoryFactoryCPU)
    {
        auto eulerParameters = alsfvm::make_shared<equation::euler::EulerParameters>();

        simulatorParameters.setEquationParameters(eulerParameters);
    }

    void makeReconstruction(const std::string name, size_t newNx) {
        nx = newNx;
        nz = 1;
        ny = 1;
        
        grid = Grid({ 0, 0, 0 }, { 1, 1, 1 }, ivec3(nx, ny, nz));

        makeReconstruction(name);
    }

    void makeReconstruction(const std::string& name) {
        wenoCUDA = reconstructionFactory.createReconstruction(name, "euler3", simulatorParameters, memoryFactory, grid, deviceConfiguration);

        ngx = wenoCUDA->getNumberOfGhostCells();
        ngy = wenoCUDA->getNumberOfGhostCells();

        conserved = volumeFactory.createConservedVolume(nx, ny, nz, wenoCUDA->getNumberOfGhostCells());
        left = volumeFactory.createConservedVolume(nx, ny, nz, wenoCUDA->getNumberOfGhostCells());
        right = volumeFactory.createConservedVolume(nx, ny, nz, wenoCUDA->getNumberOfGhostCells());

        conservedCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, wenoCUDA->getNumberOfGhostCells());
        rightCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, wenoCUDA->getNumberOfGhostCells());
        leftCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, wenoCUDA->getNumberOfGhostCells());

        conserved->makeZero();
    }

};

TEST_F(CUDAWenoTest, ConstantZeroTestSecondOrder) {
    makeReconstruction("weno2");
 
    wenoCUDA->performReconstruction(*conserved, 0, 0, *left, *right);

	right->copyTo(*rightCPU);
	left->copyTo(*leftCPU);
    for_each_cell_index(*leftCPU, [&](size_t middle) {
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
    }, {ngx-1, ngy, ngz}, {ngx-1, ngy, ngz});
}
TEST_F(CUDAWenoTest, ConstantOneTestSecondOrder) {
    makeReconstruction("weno2");
    for_each_cell_index(*conservedCPU, [&](size_t index) {
		conservedCPU->getScalarMemoryArea("rho")->getPointer()[index] = 1;
		conservedCPU->getScalarMemoryArea("mx")->getPointer()[index] = 1;
		conservedCPU->getScalarMemoryArea("my")->getPointer()[index] = 1;
		conservedCPU->getScalarMemoryArea("mz")->getPointer()[index] = 1;
		conservedCPU->getScalarMemoryArea("E")->getPointer()[index] = 10;

	});

	conservedCPU->copyTo(*conserved);


	wenoCUDA->performReconstruction(*conserved, 0, 0, *left, *right);

	left->copyTo(*leftCPU);
	right->copyTo(*rightCPU);

    for_each_cell_index(*left, [&](size_t middle) {
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
    }, {ngx-1, ngy, ngz}, {ngx-1, ngy, ngz});
}


TEST_F(CUDAWenoTest, ReconstructionSimple) {
	makeReconstruction("weno2");
   
	for_each_cell_index(*conservedCPU, [&](size_t index) {
		// fill some dummy data
		conservedCPU->getScalarMemoryArea("rho")->getPointer()[index] = 1;
		conservedCPU->getScalarMemoryArea("mx")->getPointer()[index] = 1;
		conservedCPU->getScalarMemoryArea("my")->getPointer()[index] = 1;
		conservedCPU->getScalarMemoryArea("mz")->getPointer()[index] = 1;
		conservedCPU->getScalarMemoryArea("E")->getPointer()[index] = 10;

	});


	// This is the main ingredient:
    conservedCPU->getScalarMemoryArea("rho")->getPointer()[4*(nx+4)+1] = 2;
    conservedCPU->getScalarMemoryArea("rho")->getPointer()[4*(nx+4)+2] = 0;
    conservedCPU->getScalarMemoryArea("rho")->getPointer()[4*(nx+4)+3] = 1;

	conservedCPU->copyTo(*conserved);


	wenoCUDA->performReconstruction(*conserved, 0, 0, *left, *right);
	cudaDeviceSynchronize();
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

	right->copyTo(*rightCPU);
	left->copyTo(*leftCPU);

    ASSERT_NEAR(omega0 * right1 + omega1 * right2, rightCPU->getScalarMemoryArea("rho")->getPointer()[4*(nx+4)+2], 1e-8);
    ASSERT_NEAR(omega0Tilde * left1 + omega1Tilde * left2, leftCPU->getScalarMemoryArea("rho")->getPointer()[4*(nx+4)+2], 1e-8);

}
