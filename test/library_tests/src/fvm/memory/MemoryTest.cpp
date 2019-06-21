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

#include "alsutils/config.hpp"
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

    alsfvm::cuda::CudaMemory<alsfvm::real> memory(size);
    memory.copyFromHost(buffer.data(), buffer.size());

    alsfvm::cuda::CudaMemory<alsfvm::real> memory2(size);
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

    alsfvm::cuda::CudaMemory<alsfvm::real> memory(size);
    memory.copyFromHost(buffer.data(), buffer.size());

    alsfvm::cuda::CudaMemory<alsfvm::real> memory2(size);
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


    alsfvm::cuda::CudaMemory<alsfvm::real> memory(size);
    memory.copyFromHost(buffer.data(), buffer.size());


    alsfvm::cuda::CudaMemory<alsfvm::real> memory2(size);

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
TEST(CudaMemoryTest, AdditionScalarTest) {
    size_t size = 10;

    std::vector<alsfvm::real> buffer(size);

    for (size_t i = 0; i < size; i++) {
        buffer[i] = i;
    }

    alsfvm::cuda::CudaMemory<alsfvm::real> memory(size);
    memory.copyFromHost(buffer.data(), buffer.size());

    const alsfvm::real scalar = 42.42;
    memory += scalar;

    std::vector<alsfvm::real> bufferOut1(size);

    memory.copyToHost(bufferOut1.data(), size);

    auto data1 = bufferOut1.data();

    for (size_t i = 0; i < size; i++) {
        ASSERT_EQ(i + scalar, data1[i]);
    }
}


TEST(CudaMemoryTest, MultiplyScalarTest) {
    size_t size = 10;

    std::vector<alsfvm::real> buffer(size);

    for (size_t i = 0; i < size; i++) {
        buffer[i] = i;
    }

    alsfvm::cuda::CudaMemory<alsfvm::real> memory(size);
    memory.copyFromHost(buffer.data(), buffer.size());

    const alsfvm::real scalar = 42.42;
    memory *= scalar;

    std::vector<alsfvm::real> bufferOut1(size);

    memory.copyToHost(bufferOut1.data(), size);

    auto data1 = bufferOut1.data();

    for (size_t i = 0; i < size; i++) {
        ASSERT_EQ(i * scalar, data1[i]);
    }
}

TEST(CudaMemoryTest, DivideScalarTest) {
    size_t size = 10;

    std::vector<alsfvm::real> buffer(size);

    for (size_t i = 0; i < size; i++) {
        buffer[i] = i;
    }

    alsfvm::cuda::CudaMemory<alsfvm::real> memory(size);
    memory.copyFromHost(buffer.data(), buffer.size());

    const alsfvm::real scalar = 42.42;
    memory /= scalar;

    std::vector<alsfvm::real> bufferOut1(size);

    memory.copyToHost(bufferOut1.data(), size);

    auto data1 = bufferOut1.data();

    for (size_t i = 0; i < size; i++) {
        ASSERT_FLOAT_EQ(i / scalar, data1[i]);
    }
}

TEST(CudaMemoryTest, SubtractScalarTest) {
    size_t size = 10;

    std::vector<alsfvm::real> buffer(size);

    for (size_t i = 0; i < size; i++) {
        buffer[i] = i;
    }

    alsfvm::cuda::CudaMemory<alsfvm::real> memory(size);
    memory.copyFromHost(buffer.data(), buffer.size());

    const alsfvm::real scalar = 42.42;
    memory -= scalar;

    std::vector<alsfvm::real> bufferOut1(size);

    memory.copyToHost(bufferOut1.data(), size);

    auto data1 = bufferOut1.data();

    for (size_t i = 0; i < size; i++) {
        ASSERT_EQ(i - scalar, data1[i]);
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
        memory.getPointer() + size,
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
        buffer[i] = i + 1;
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
        ASSERT_EQ(alsfvm::real(i + 1) / alsfvm::real(42), data1[i]);
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
        ASSERT_EQ(int(i + 1) - 42, data1[i]);
    }
}


TEST(HostMemoryTest, AdditionScalarTest) {
    size_t size = 10;

    std::vector<alsfvm::real> buffer(size);

    for (size_t i = 0; i < size; i++) {
        buffer[i] = i;
    }

    alsfvm::memory::HostMemory<alsfvm::real> memory(size);
    memory.copyFromHost(buffer.data(), buffer.size());

    const alsfvm::real scalar = 42.42;
    memory += scalar;

    std::vector<alsfvm::real> bufferOut1(size);

    memory.copyToHost(bufferOut1.data(), size);

    auto data1 = bufferOut1.data();

    for (size_t i = 0; i < size; i++) {
        ASSERT_EQ(i + scalar, data1[i]);
    }
}


TEST(HostMemoryTest, MultiplyScalarTest) {
    size_t size = 10;

    std::vector<alsfvm::real> buffer(size);

    for (size_t i = 0; i < size; i++) {
        buffer[i] = i;
    }

    alsfvm::memory::HostMemory<alsfvm::real> memory(size);
    memory.copyFromHost(buffer.data(), buffer.size());

    const alsfvm::real scalar = 42.42;
    memory *= scalar;

    std::vector<alsfvm::real> bufferOut1(size);

    memory.copyToHost(bufferOut1.data(), size);

    auto data1 = bufferOut1.data();

    for (size_t i = 0; i < size; i++) {
        ASSERT_EQ(i * scalar, data1[i]);
    }
}

TEST(HostMemoryTest, DivideScalarTest) {
    size_t size = 10;

    std::vector<alsfvm::real> buffer(size);

    for (size_t i = 0; i < size; i++) {
        buffer[i] = i;
    }

    alsfvm::memory::HostMemory<alsfvm::real> memory(size);
    memory.copyFromHost(buffer.data(), buffer.size());

    const alsfvm::real scalar = 42.42;
    memory /= scalar;

    std::vector<alsfvm::real> bufferOut1(size);

    memory.copyToHost(bufferOut1.data(), size);

    auto data1 = bufferOut1.data();

    for (size_t i = 0; i < size; i++) {
        ASSERT_EQ(i / scalar, data1[i]);
    }
}

TEST(HostMemoryTest, SubtractScalarTest) {
    size_t size = 10;

    std::vector<alsfvm::real> buffer(size);

    for (size_t i = 0; i < size; i++) {
        buffer[i] = i;
    }

    alsfvm::memory::HostMemory<alsfvm::real> memory(size);
    memory.copyFromHost(buffer.data(), buffer.size());

    const alsfvm::real scalar = 42.42;
    memory -= scalar;

    std::vector<alsfvm::real> bufferOut1(size);

    memory.copyToHost(bufferOut1.data(), size);

    auto data1 = bufferOut1.data();

    for (size_t i = 0; i < size; i++) {
        ASSERT_EQ(i - scalar, data1[i]);
    }
}

TEST(HostMemoryTest, 2DArrayTest) {
    size_t nx = 16;
    size_t ny = 8;
    size_t nz = 4;

    alsfvm::memory::HostMemory<alsfvm::real> memory(nx, ny, nz);

    ASSERT_EQ(memory.getSizeX(), nx);
    ASSERT_EQ(memory.getSizeY(), ny);
    ASSERT_EQ(memory.getSizeZ(), nz);

    auto data = memory.getPointer();

    for (size_t i = 0; i < nz; i++) {
        for (size_t j = 0; j < ny; j++) {
            for (size_t k = 0; k < nx; k++) {

                memory.at(k, j, i) = calculateIndex(k, j, i, nx, ny);
            }
        }
    }

    for (size_t i = 0; i < nz; i++) {
        for (size_t j = 0; j < ny; j++) {
            for (size_t k = 0; k < nx; k++) {

                ASSERT_EQ(memory.at(k, j, i), data[calculateIndex(k, j, i, nx, ny)]);
                ASSERT_EQ(memory.at(k, j, i), calculateIndex(k, j, i, nx, ny));
            }
        }
    }
}

TEST(HostMemoryTest, ViewTest) {
    size_t nx = 16;
    size_t ny = 8;
    size_t nz = 4;

    alsfvm::memory::HostMemory<alsfvm::real> memory(nx, ny, nz);

    auto view = memory.getView();

    view.at(1, 2, 3) = 4;

    const auto& constMemory = memory;

    auto constView = constMemory.getView();

    ASSERT_EQ(4, view.at(1, 2, 3));
    ASSERT_EQ(4, constView.at(1, 2, 3));
}

TEST(HostMemoryTest, ViewIndexTest) {
    size_t nx = 16;
    size_t ny = 8;
    size_t nz = 4;

    alsfvm::memory::HostMemory<alsfvm::real> memory(nx, ny, nz);

    auto view = memory.getView();

    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t index = z * nx * ny + y * nx + x;
                ASSERT_EQ(index, view.index(x, y, z))
                        << "Wrong index at (" << x << ", " << y << ", " << z << ")";
            }
        }
    }
}

TEST(HostMemoryTest, ViewIndexTest2D) {
    size_t nx = 16;
    size_t ny = 8;
    size_t nz = 1;

    alsfvm::memory::HostMemory<alsfvm::real> memory(nx, ny, nz);

    auto view = memory.getView();

    for (size_t z = 0; z < nz; z++) {
        for (size_t y = 0; y < ny; y++) {
            for (size_t x = 0; x < nx; ++x) {
                size_t index = z * nx * ny + y * nx + x;
                ASSERT_EQ(index, view.index(x, y, z))
                        << "Wrong index at (" << x << ", " << y << ", " << z << ")";
            }
        }
    }
}


