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
      CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));
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
	  cudaDeviceReset();
	  cudaThreadExit();

	  // Round robin kind of allocation (here we assume the mpi nodes gets assigned 
	  // rank in this way)
	  const int device = mpiRank % deviceCount;
	  ALSVINN_LOG(INFO, "Setting CUDA device to " << device);
	  CUDA_SAFE_CALL(cudaSetDevice(device));

	  // Make sure it worked
	  int currentDevice = -1;
	  CUDA_SAFE_CALL(cudaGetDevice(&currentDevice));
	  ALSVINN_LOG(INFO, "Current CUDA device is " << currentDevice);
	
	  if (currentDevice != device) {
	    ALSVINN_LOG(WARNING, "Could not update current device (" << device << ", " << currentDevice << ")");
	  }
	}
      }
    }
  }
}
#else

namespace alsutils {
  namespace mpi {
    //    inline void setCudaDevice() {
      // dummy function for when we do not have cuda

    //}
  }
}
#endif
