#include "alsfvm/cuda/CudaMemory.hpp"
#include "gtest/gtest.h"
#include <vector>

TEST(CudaMemoryTest, InitializeTest) {
	size_t size = 10;
	alsfvm::cuda::CudaMemory memory(size);
	ASSERT_EQ(size, memory.getSize());
}

TEST(CudaMemoryTest, TransferTest) {
	size_t size = 10;

	std::vector<char> buffer(size);

	for (size_t i = 0; i < size; i++) {
		buffer[i] = i;
	}

	alsfvm::cuda::CudaMemory memory(size);

	memory.copyFromHost(buffer.data(), buffer.size());

	std::vector<char> bufferOut(size, 0);

	memory.copyToHost(bufferOut.data(), bufferOut.size());

	for (size_t i = 0; i < size; i++) {
		ASSERT_EQ(bufferOut[i], buffer[i]);
	}
}