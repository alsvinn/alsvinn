#pragma once
#ifdef ALSVINN_HAVE_CUDA
#include <cuda_runtime.h>
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
      cudaGetDeviceCount(&deviceCount);
      
      if (deviceCount > 1) {
	int mpiRank = -1;
	int mpiSize = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

	const int device = mpiRank / deviceCount;

	cudaSetDevice(device);

      }
    }
  }
}
#else

namespace alsutils {
  namespace mpi {
    inline void setCudaDevice() {
      // dummy function for when we do not have cuda

    }
  }
}
#endif
