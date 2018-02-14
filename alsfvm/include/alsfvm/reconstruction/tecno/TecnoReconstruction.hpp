#pragma once
#include "alsfvm/volume/Volume.hpp"

namespace alsfvm {
namespace reconstruction {
namespace tecno {

//! Special abstract super class for reconstruction for ENO.
//!
//! The reason we need a different class than for normal reconstruction
//! is that the input left and right values are a priori different.
//!
//! In other words, for tecno we reconstruct with
//!
//!    \f[u^l_i = R_{i+1/2}u_{i}\f]
//!    \f[u^r_i = R_{i-1/2}u_{i}\f]
//!
//! The reconstructions should be compatible with the Tecno paper
//!
//! Fjordholm, U. S., Mishra, S., & Tadmor, E. (2012). Arbitrarily high-order accurate entropy stable essentially nonoscillatory schemes for systems of conservation laws, 50(2), 544–573.
//!
//! See http://www.cscamm.umd.edu/people/faculty/tadmor/pub/TV+entropy/Fjordholm_Mishra_Tadmor_SINUM2012.pdf
class TecnoReconstruction {
public:
    virtual ~TecnoReconstruction() {}

    //! Applies the reconstruction.
    //!
    //! @param[in] leftInput the left values to use for reconstruction
    //! @param[in] rightInput the right values to use for reconstruction
    //! @param[in] direction the direction (0=x, 1=y, 2=y)
    //! @param[out] leftOutput at the end, should contain reconstructed values
    //! @param[out] rightOutput at the end, should contain the reconstructed values
    virtual void performReconstruction(const volume::Volume& leftInput,
        const volume::Volume& rightInput,
        size_t direction,
        volume::Volume& leftOutput,
        volume::Volume& rightOutput) = 0;

    virtual size_t getNumberOfGhostCells() const = 0;
};
} // namespace tecno
} // namespace reconstruction
} // namespace alsfvm
