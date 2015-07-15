
#include "gtest/gtest.h"
#include <vector>

#include "alsfvm/memory/HostMemory.hpp"
#ifdef ALSVINN_HAVE_CUDA
#include "alsfvm/cuda/CudaMemory.hpp"
TEST(CudaMemoryTest, InitializeTest) {
	size_t size = 10;
    alsfvm::cuda::CudaMemory<alsfvm::real> memory(size);
	ASSERT_EQ(size, memory.getSize());
}

TEST(CudaMemoryTest, TransferTest) {
	size_t size = 10;

    std::vector<alsfvm::real> buffer(size);

	for (size_t i = 0; i < size; i++) {
		buffer[i] = i;
	}

    alsfvm::cuda::CudaMemory<alsfvm::real> memory(size);

	memory.copyFromHost(buffer.data(), buffer.size());

    std::vector<alsfvm::real> bufferOut(size, 0);

	memory.copyToHost(bufferOut.data(), bufferOut.size());

	for (size_t i = 0; i < size; i++) {
		ASSERT_EQ(bufferOut[i], buffer[i]);
	}
}
#endif

TEST(HostMemoryTest, InitializeTest) {
    size_t size = 10;
    alsfvm::memory::HostMemory<alsfvm::real> memory(size);
    ASSERT_EQ(size, memory.getSize());
}

TEST(HostMemoryTest, TransferTest) {
    size_t size = 10;

    std::vector<alsfvm::real> buffer(size);

    for (size_t i = 0; i < size; i++) {
        buffer[i] = i;
    }

    alsfvm::memory::HostMemory<alsfvm::real> memory(size);

    memory.copyFromHost(buffer.data(), buffer.size());

    std::vector<alsfvm::real> bufferOut(size, 0);

    memory.copyToHost(bufferOut.data(), bufferOut.size());

    for (size_t i = 0; i < size; i++) {
        ASSERT_EQ(bufferOut[i], buffer[i]);
    }
}

TEST(HostMemoryTest, OnHostTest) {
    size_t size = 10;

    std::vector<alsfvm::real> buffer(size);

    for (size_t i = 0; i < size; i++) {
        buffer[i] = i;
    }

    alsfvm::memory::HostMemory<alsfvm::real> memory(size);
    memory.copyFromHost(buffer.data(), buffer.size());
    ASSERT_TRUE(memory.isOnHost());

    std::vector<alsfvm::real> bufferOut(size, 0);

    std::copy(memory.getPointer(),
              memory.getPointer()+size,
              bufferOut.begin());

    for (size_t i = 0; i < size; i++) {
        ASSERT_EQ(bufferOut[i], buffer[i]);
    }
}
