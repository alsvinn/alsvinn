#pragma once
#include "alsfvm/functional/Functional.hpp"

namespace alsfvm {
namespace functional {

//!
//! This just dumps the solution at the current time
//!
class IdentityCUDA : public Functional {
public:

    //! The following parameters are accepted through parameters
    //!
    //!    Name      | Description
    //!    ----------|-------------
    //!    variables | the variables to compute for (space separated)
    IdentityCUDA(const Parameters& parameters);

    //! Computes the operator value on the givne input data
    //!
    //! @note In order to support time integration, the result should be
    //!       added to conservedVolumeOut and extraVolumeOut, not overriding
    //!       it.
    //!
    //! @param[out] conservedVolumeOut at the end, should have the contribution
    //!             of the functional for the conservedVariables
    //!
    //! @param[out] extraVolumeOut at the end, should have the contribution
    //!             of the functional for the extraVariables
    //!
    //! @param[in] conservedVolumeIn the state of the conserved variables
    //!
    //! @param[in] extraVolumeIn the state of the extra volume
    //!
    //! @param[in] weight the current weight to be applied to the functional. Ie, the functional should compute
    //!                   \code{.cpp}
    //!                   conservedVolumeOut += weight + f(conservedVolumeIn)
    //!                   \endcode
    //!
    //! @param[in] grid the grid to use
    //!
    virtual void operator()(volume::Volume& conservedVolumeOut,
        volume::Volume& extraVolumeOut,
        const volume::Volume& conservedVolumeIn,
        const volume::Volume& extraVolumeIn,
        const real weight,
        const grid::Grid& grid
    ) override;

    //! Returns grid.getDimensions()
    virtual ivec3 getFunctionalSize(const grid::Grid& grid) const override;


private:
    std::vector<std::string> variables;

};
} // namespace functional
} // namespace alsfvm
