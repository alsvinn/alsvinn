
#pragma once
#include "alsfvm/functional/Functional.hpp"
namespace alsfvm {
namespace functional {

//! Computes the bounded variation of the given solution
class BoundedVariation : public Functional {
public:
    //! The following parameters are accepted through parameters
    //!
    //!    Name      | Description
    //!    ----------|-------------
    //!    degree    | The degree
    BoundedVariation(const Parameters& parameters);

    //! Computes the operator value on the givne input data
    //!
    //! @note In order to support time integration, the result should be
    //!       added to conservedVolumeOut and extraVolumeOut, not overriding
    //!       it.
    //!
    //! @param[out] conservedVolumeOut at the end, should have the contribution
    //!             of the functional for the conservedVariables
    //!
    //! @param[in] conservedVolumeIn the state of the conserved variables
    //!
    //!
    //! @param[in] weight the current weight to be applied to the functional. Ie, the functional should compute
    //!                   \code{.cpp}
    //!                   conservedVolumeOut += weight + f(conservedVolumeIn)
    //!                   \endcode
    //! @param[in] grid the current grid
    //!
    virtual void operator()(volume::Volume& conservedVolumeOut,
        const volume::Volume& conservedVolumeIn,
        const real weight,
        const grid::Grid& grid
    ) override;

    //! Returns ivec3{1,1,1} -- we only need one element to represent this functional
    virtual ivec3 getFunctionalSize(const grid::Grid& grid) const override;


    virtual std::string getPlatformToAllocateOn(const std::string& platform) const
    override
    ;
private:
    const int degree = 1;

};
} // namespace functional
} // namespace alsfvm
