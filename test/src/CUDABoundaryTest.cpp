#include <gtest/gtest.h>
#include "alsfvm/boundary/BoundaryFactory.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/cuda/cuda_utils.hpp"
using namespace alsfvm::boundary;
using namespace alsfvm;
using namespace alsfvm::volume;

class CUDABoundaryTest : public ::testing::Test {
public:
	const size_t nx;
	const size_t ny;
	const size_t nz;

	std::string equation;
	alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration;
	grid::Grid grid;
	alsfvm::shared_ptr<memory::MemoryFactory> memoryFactory;
	volume::VolumeFactory volumeFactory;
	boundary::BoundaryFactory boundaryFactory;

	CUDABoundaryTest()
        :
		nx(10), ny(10), nz(1),
        equation("euler"),
        deviceConfiguration(new DeviceConfiguration("cuda")),
		grid(rvec3(0, 0, 0), rvec3(1, 1, 1), ivec3(nx, ny, nz)),
		memoryFactory(new memory::MemoryFactory(deviceConfiguration)),
		volumeFactory(equation, memoryFactory), boundaryFactory("neumann", deviceConfiguration)
	{

	}
};


TEST_F(CUDABoundaryTest, TestOnes) {
	const real baseValue = 123.123;
	const real size = nx * ny *nz;
	std::cout << size << std::endl;
	auto volume = volumeFactory.createConservedVolume(nx, ny, nz, 0);
	std::vector<real> inputVector(size);
	for (size_t z = 0; z < nz; ++z) {
		for (size_t y = 0; y < ny; ++y) {
			for (size_t x = 0; x < nx; ++x) {
				size_t index = z * nx * ny + y * nx + x;
				if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1) {
					inputVector[index] = 1;
				}
				else {
					inputVector[index] = baseValue;
				}
			}
		}
	}
	volume->getScalarMemoryArea(0)->copyFromHost(inputVector.data(), inputVector.size());
	std::vector<real> result(size, 0);
	volume->getScalarMemoryArea(0)->copyToHost(result.data(), result.size());
	for (size_t z = 0; z < nz; ++z) {
		for (size_t y = 0; y < ny; ++y) {
			for (size_t x = 0; x < nx; ++x) {
				size_t index = z * nx * ny + y * nx + x;
				const size_t x1 = index % nx;
				const size_t y1 = (index / nx) % ny;
				const size_t z1 = (index / nx) / ny;
				ASSERT_EQ(x1, x);
				ASSERT_EQ(y1, y);
				ASSERT_EQ(z1, z);
				assert(index < size);
				if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 ) {
					ASSERT_EQ(1, result[index]) << "Mismatch at index (" << x << ", " << y << ", " << z << "), index = " << index << std::endl ;
				}
				else {
					ASSERT_EQ(baseValue, result[index]);
				}
			}
		}
	}
	auto boundary = boundaryFactory.createBoundary(1);
	boundary->applyBoundaryConditions(*volume, grid);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	
	volume->getScalarMemoryArea(0)->copyToHost(result.data(), result.size());

	for (size_t z = 0; z < nz; ++z) {
		for (size_t y = 0; y < ny; ++y) {
			for (size_t x = 0; x < nx; ++x) {
				size_t index = z * nx * ny + y * nx + x;
				if ((x == 0 && y == 0 && z == 0) || (x == nx - 1 && y == ny - 1) || (x == 0 && y == ny - 1) || (x == nx - 1 && y == 0)) {
					continue;
				}
				ASSERT_EQ(baseValue, result[index]) << "Mismatch at index (" << x << ", " << y << ", " << z << "), index = " << index << std::endl;
			}
		}
	}
}
