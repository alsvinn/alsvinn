
#include "gtest/gtest.h"
#include <vector>

#include "alsfvm/memory/MemoryFactory.hpp"
#ifdef ALSVINN_HAVE_CUDA
#include "alsfvm/cuda/CudaMemory.hpp"
TEST(CudaMemoryFactoryTest, CreateCudaMemoryArea) {
	auto deviceConfiguration = std::make_shared<alsfvm::DeviceConfiguration>();
	const std::string memoryName = "CudaMemory";
	alsfvm::memory::MemoryFactory factory(memoryName, deviceConfiguration);

	size_t size = 10;
	auto memory = factory.createScalarMemory(size);
	ASSERT_EQ(size, memory->getSize());

	auto cudaMemory = std::dynamic_pointer_cast<alsfvm::cuda::CudaMemory <alsfvm::real >> (memory);
	ASSERT_TRUE(!!cudaMemory);
}

#endif

#include "alsfvm/memory/HostMemory.hpp"
TEST(HostMemoryFactoryTest, CreateHostMemoryArea) {
	auto deviceConfiguration = std::make_shared<alsfvm::DeviceConfiguration>();
	const std::string memoryName = "HostMemory";
	alsfvm::memory::MemoryFactory factory(memoryName, deviceConfiguration);

	size_t size = 10;
	auto memory = factory.createScalarMemory(size);
	ASSERT_EQ(size, memory->getSize());


	auto hostMemory = std::dynamic_pointer_cast<alsfvm::memory::HostMemory<alsfvm::real>>(memory);
	ASSERT_TRUE(!!hostMemory);
}