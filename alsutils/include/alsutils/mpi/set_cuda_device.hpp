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

#pragma once
#ifdef ALSVINN_HAVE_CUDA
#define ENV_LOCAL_RANK "OMPI_COMM_WORLD_LOCAL_RANK"
#include <cuda_runtime.h>
#include "alsutils/log.hpp"
#include "alsutils/cuda/cuda_safe_call.hpp"
#include <iostream>
#include <cstdlib>

namespace alsutils {
namespace mpi {
//! Tries to set the cuda device correctly in the setting
//! where there are more than one device per node
//! @note This assumes we are given a "smart setup",
//!       meaning that cuda_device_count * number_of_nodes = number_of_processes
//!
//!
inline void setCudaDevice() {
    int deviceCount = -1;

    try {
        CUDA_SAFE_CALL_SILENT(cudaGetDeviceCount(&deviceCount));
        ALSVINN_LOG(INFO, "Number of GPUs on node: " << deviceCount);

        if (deviceCount > 1) {


            // We need to do this before we call MPI_Init
            // (I think at least, hint about this found in
            // https://devtalk.nvidia.com/default/topic/752046/teaching-and-curriculum-support/multi-gpu-system-running-mpi-cuda-/
            // )
            //
            // Testing revealed this to "probably" be necessary. (2018-05-03 on the Leonhard Cluster at ETHZ)
            const char* rankAsString = std::getenv(ENV_LOCAL_RANK);

            if (rankAsString) {

                int mpiRank = std::atoi(rankAsString);

                // Reset the state completely (This may not be neccessary)
                CUDA_SAFE_CALL_SILENT(cudaDeviceReset());
                CUDA_SAFE_CALL_SILENT(cudaThreadExit());

                // Round robin kind of allocation (here we assume the mpi nodes gets assigned
                // rank in this way)
                const int device = mpiRank % deviceCount;
                ALSVINN_LOG(INFO, "Setting CUDA device to " << device);
                CUDA_SAFE_CALL_SILENT(cudaSetDevice(device));

                // Make sure it worked
                int currentDevice = -1;
                CUDA_SAFE_CALL_SILENT(cudaGetDevice(&currentDevice));
                ALSVINN_LOG(INFO, "Current CUDA device is " << currentDevice);

                if (currentDevice != device) {
                    ALSVINN_LOG(WARNING, "Could not update current device (" << device << ", " <<
                        currentDevice << ")");
                }
            }
        }
    } catch (std::runtime_error& e) {
        ALSVINN_LOG(WARNING,
            "(Ignore this if you are not running with CUDA). Failed setting CUDA GPU: " <<
            e.what());
    }
}
}
}
#else

namespace alsutils {
namespace mpi {
inline void setCudaDevice() {
    //  dummy function for when we do not have cuda

}
}
}
#endif
