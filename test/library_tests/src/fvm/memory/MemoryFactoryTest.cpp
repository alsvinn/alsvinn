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


#include "gtest/gtest.h"
#include <vector>

#include "alsfvm/memory/MemoryFactory.hpp"
#ifdef ALSVINN_HAVE_CUDA
#include "alsfvm/cuda/CudaMemory.hpp"
TEST(CudaMemoryFactoryTest, CreateCudaMemoryArea) {
    auto deviceConfiguration =
        alsfvm::make_shared<alsfvm::DeviceConfiguration>("cuda");
    alsfvm::memory::MemoryFactory factory(deviceConfiguration);


    size_t nx = 10;
    size_t ny = 10;
    size_t nz = 10;
    auto memory = factory.createScalarMemory(nx, ny, nz);
    ASSERT_EQ(nx * ny * nz, memory->getSize());

    auto cudaMemory =
        alsfvm::dynamic_pointer_cast<alsfvm::cuda::CudaMemory <alsfvm::real >> (memory);
    ASSERT_TRUE(!!cudaMemory);
}

#endif

#include "alsfvm/memory/HostMemory.hpp"
TEST(HostMemoryFactoryTest, CreateHostMemoryArea) {
    auto deviceConfiguration =
        alsfvm::make_shared<alsfvm::DeviceConfiguration>("cpu");

    alsfvm::memory::MemoryFactory factory(deviceConfiguration);


    size_t nx = 10;
    size_t ny = 10;
    size_t nz = 10;
    auto memory = factory.createScalarMemory(nx, ny, nz);
    ASSERT_EQ(nx * ny * nz, memory->getSize());

    ASSERT_EQ(nx, memory->getSizeX());
    ASSERT_EQ(ny, memory->getSizeY());
    ASSERT_EQ(nz, memory->getSizeZ());

    auto hostMemory =
        alsfvm::dynamic_pointer_cast<alsfvm::memory::HostMemory<alsfvm::real>>(memory);
    ASSERT_TRUE(!!hostMemory);
}
