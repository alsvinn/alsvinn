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

#include <gtest/gtest.h>

#ifdef ALSVINN_USE_MPI
    #include <mpi.h>
#endif
#ifdef ALSVINN_HAVE_CUDA
    #include <cuda_runtime.h>
#endif
#include <cstdlib>
int main(int argc, char** argv) {
#ifdef ALSVINN_USE_MPI
#ifdef DALSVINN_HAS_GPU_DIRECT
    setenv("MPICH_RDMA_ENABLED_CUDA", "1", 1);
    setenv("MV2_USE_CUDA", "1", 1);
#endif
    MPI_Init(NULL, NULL);
#ifdef DALSVINN_HAS_GPU_DIRECT

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double* a;
    cudaMalloc(&a, sizeof(a));

    if (size > 1) {
        MPI_Request requestSend;
        MPI_Isend(a, 1, MPI_DOUBLE, (rank + 1) % size, 0, MPI_COMM_WORLD,
            &requestSend);
        MPI_Request requestReceive;
        MPI_Irecv(a, 1, MPI_DOUBLE, (rank + 1) % size, 0, MPI_COMM_WORLD,
            &requestReceive);
        MPI_Wait(&requestSend, MPI_STATUS_IGNORE);
        MPI_Wait(&requestReceive, MPI_STATUS_IGNORE);

    }

#endif
#endif



    ::testing::InitGoogleTest(&argc, argv);

    auto exitValue = RUN_ALL_TESTS();
    MPI_Finalize();
    return exitValue;
}
