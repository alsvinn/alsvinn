#pragma once
#include "alsutils/config.hpp"
#ifdef ALSVINN_HAVE_CUDA
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include "alsutils/cuda/cuda_safe_call.hpp"

#endif
namespace alsutils {
namespace cuda {
//! Returns the current active GPU id, or -1 if cuda is disabled
inline int getCurrentGPUId() {
    int deviceID = -1;
#ifdef ALSVINN_HAVE_CUDA
    CUDA_SAFE_CALL(cudaGetDevice(&deviceID));
#endif

    return deviceID;
}
}
}
