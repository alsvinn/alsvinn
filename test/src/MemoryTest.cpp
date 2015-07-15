
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


TEST(CudaMemoryTest, AdditionTest) {
	size_t size = 10;

	std::vector<alsfvm::real> buffer(size);

	for (size_t i = 0; i < size; i++) {
		buffer[i] = i;
	}

	alsfvm::memory::HostMemory<alsfvm::real> memory(size);
	memory.copyFromHost(buffer.data(), buffer.size());

	alsfvm::memory::HostMemory<alsfvm::real> memory2(size);
	memory2.copyFromHost(buffer.data(), buffer.size());

	memory += memory2;

	std::vector<alsfvm::real> bufferOut1(size);
	std::vector<alsfvm::real> bufferOut2(size);
	memory.copyToHost(bufferOut1.data(), size);
	memory2.copyToHost(bufferOut2.data(), size);
	auto data1 = bufferOut1.data();
	auto data2 = bufferOut2.data();
	for (size_t i = 0; i < size; i++) {
		ASSERT_EQ(i, data2[i]);
		ASSERT_EQ(2 * i, data1[i]);
	}
}


TEST(CudaMemoryTest, MultiplyTest) {
	size_t size = 10;

	std::vector<alsfvm::real> buffer(size);

	for (size_t i = 0; i < size; i++) {
		buffer[i] = i;
	}

	alsfvm::memory::HostMemory<alsfvm::real> memory(size);
	memory.copyFromHost(buffer.data(), buffer.size());

	alsfvm::memory::HostMemory<alsfvm::real> memory2(size);
	memory2.copyFromHost(buffer.data(), buffer.size());

	memory *= memory2;

	std::vector<alsfvm::real> bufferOut1(size);
	std::vector<alsfvm::real> bufferOut2(size);
	memory.copyToHost(bufferOut1.data(), size);
	memory2.copyToHost(bufferOut2.data(), size);
	auto data1 = bufferOut1.data();
	auto data2 = bufferOut2.data();
	for (size_t i = 0; i < size; i++) {
		ASSERT_EQ(i, data2[i]);
		ASSERT_EQ(i * i, data1[i]);
	}
}

TEST(CudaMemoryTest, DivideTest) {
	size_t size = 10;

	std::vector<alsfvm::real> buffer(size);

	for (size_t i = 0; i < size; i++) {
		buffer[i] = i + 1;
	}


	alsfvm::memory::HostMemory<alsfvm::real> memory(size);
	memory.copyFromHost(buffer.data(), buffer.size());


	alsfvm::memory::HostMemory<alsfvm::real> memory2(size);

	std::vector<alsfvm::real> buffer2(size, 42);
	memory2.copyFromHost(buffer2.data(), buffer2.size());

	memory /= memory2;

	std::vector<alsfvm::real> bufferOut1(size);
	std::vector<alsfvm::real> bufferOut2(size);
	memory.copyToHost(bufferOut1.data(), size);
	memory2.copyToHost(bufferOut2.data(), size);
	auto data1 = bufferOut1.data();
	auto data2 = bufferOut2.data();
	for (size_t i = 0; i < size; i++) {
		ASSERT_EQ(42, data2[i]);
		ASSERT_EQ(alsfvm::real(i + 1) / alsfvm::real(42), data1[i]);
	}
}

TEST(CudaMemoryTest, SubtractTest) {
	size_t size = 10;

	std::vector<alsfvm::real> buffer(size);

	for (size_t i = 0; i < size; i++) {
		buffer[i] = i + 1;
	}

	alsfvm::memory::HostMemory<alsfvm::real> memory(size);
	memory.copyFromHost(buffer.data(), buffer.size());

	alsfvm::memory::HostMemory<alsfvm::real> memory2(size);

	std::vector<alsfvm::real> buffer2(size, 42);
	memory2.copyFromHost(buffer2.data(), buffer2.size());

	memory -= memory2;

	auto data1 = memory.getPointer();
	auto data2 = memory2.getPointer();

	for (size_t i = 0; i < size; i++) {
		ASSERT_EQ(alsfvm::real(42), data2[i]);
		ASSERT_EQ(int(i + 1) - 42, data1[i]);
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

TEST(HostMemoryTest, AdditionTest) {
	size_t size = 10;

	std::vector<alsfvm::real> buffer(size);

	for (size_t i = 0; i < size; i++) {
		buffer[i] = i;
	}

	alsfvm::memory::HostMemory<alsfvm::real> memory(size);
	memory.copyFromHost(buffer.data(), buffer.size());
	
	alsfvm::memory::HostMemory<alsfvm::real> memory2(size);
	memory2.copyFromHost(buffer.data(), buffer.size());

	memory += memory2;

	auto data1 = memory.getPointer();
	auto data2 = memory2.getPointer();
	for (size_t i = 0; i < size; i++) {
		ASSERT_EQ(i, data2[i]);
		ASSERT_EQ(2 * i, data1[i]);
	}
}


TEST(HostMemoryTest, MultiplyTest) {
	size_t size = 10;

	std::vector<alsfvm::real> buffer(size);

	for (size_t i = 0; i < size; i++) {
		buffer[i] = i;
	}

	alsfvm::memory::HostMemory<alsfvm::real> memory(size);
	memory.copyFromHost(buffer.data(), buffer.size());

	alsfvm::memory::HostMemory<alsfvm::real> memory2(size);
	memory2.copyFromHost(buffer.data(), buffer.size());

	memory *= memory2;

	auto data1 = memory.getPointer();
	auto data2 = memory2.getPointer();
	for (size_t i = 0; i < size; i++) {
		ASSERT_EQ(i, data2[i]);
		ASSERT_EQ(i * i, data1[i]);
	}
}

TEST(HostMemoryTest, DivideTest) {
	size_t size = 10;

	std::vector<alsfvm::real> buffer(size);

	for (size_t i = 0; i < size; i++) {
		buffer[i] = i+1;
	}


	alsfvm::memory::HostMemory<alsfvm::real> memory(size);
	memory.copyFromHost(buffer.data(), buffer.size());


	alsfvm::memory::HostMemory<alsfvm::real> memory2(size);

	std::vector<alsfvm::real> buffer2(size, 42);
	memory2.copyFromHost(buffer2.data(), buffer2.size());

	memory /= memory2;

	auto data1 = memory.getPointer();
	auto data2 = memory2.getPointer();
	for (size_t i = 0; i < size; i++) {
		ASSERT_EQ(42, data2[i]);
		ASSERT_EQ(alsfvm::real(i+1)/alsfvm::real(42), data1[i]);
	}
}

TEST(HostMemoryTest, SubtractTest) {
	size_t size = 10;

	std::vector<alsfvm::real> buffer(size);

	for (size_t i = 0; i < size; i++) {
		buffer[i] = i + 1;
	}

	alsfvm::memory::HostMemory<alsfvm::real> memory(size);
	memory.copyFromHost(buffer.data(), buffer.size());

	alsfvm::memory::HostMemory<alsfvm::real> memory2(size);

	std::vector<alsfvm::real> buffer2(size, 42);
	memory2.copyFromHost(buffer2.data(), buffer2.size());

	memory -= memory2;

	auto data1 = memory.getPointer();
	auto data2 = memory2.getPointer();
	
	for (size_t i = 0; i < size; i++) {
		ASSERT_EQ(alsfvm::real(42), data2[i]);
		ASSERT_EQ(int(i+1)-42, data1[i]);
	}
}


