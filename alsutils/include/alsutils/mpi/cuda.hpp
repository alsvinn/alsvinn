#pragma once


//! Various utilities for mpi and cuda.


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
#ifdef ALSVINN_HAS_GPU_DIRECT
    return true;
#else
    return false;
#endif
}
}}
