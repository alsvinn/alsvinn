#pragma once


//! Various utilities for mpi and cuda.



//#include <mpi-ext.h>


namespace alsutils { namespace mpi {


//! Checks wether GPU direct is enabled or not
//!
//! We check according to this documentation
//!
//!    https://www.open-mpi.org/faq/?category=runcuda
//!
//! under
//!
//!   4. Can I tell at compile time or runtime whether I have CUDA-aware support?
bool hasGPUDirectSupport() {
#if 0
#if defined(MPIX_CUDA_AWARE_SUPPORT)
#if !MPIX_CUDA_AWARE_SUPPORT
    return false; // Library exist but does not have support
#endif
#else
    return false; // library does not exist
#endif /* MPIX_CUDA_AWARE_SUPPORT */

    // now for runtime checking
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    if (1 == MPIX_Query_cuda_support()) {
        return true;
    } else {
        return false;
    }

#endif /* MPIX_CUDA_AWARE_SUPPORT */
    return false;
#endif
#endif
#ifdef ALSVINN_HAS_GPU_DIRECT
    return true;
#else
    return false
#endif
}
}}
