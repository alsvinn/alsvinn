#pragma once
#include "alsfvm/functional/Functional.hpp"

namespace alsfvm { namespace functional { 

    //! This is basically the functional version of
    //! stats/StructureCube.
    //!
    //! @todo Refactor this to avoid code duplication.
    class StructureCube : public Functional {
    public:
        StructureCube(const Functional::Parameters& parameters);

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
        //! @param[in] extraVolume the state of the extra volume
        //!
        //! @param[in] weight the current weight to be applied to the functional. Ie, the functional should compute
        //!                   \code{.cpp}
        //!                   conservedVolumeOut += weight + f(conservedVolumeIn)
        //!                   \endcode
        //!
        virtual void operator()(volume::Volume& conservedVolumeOut,
                        volume::Volume& extraVolumeOut,
                        const volume::Volume& conservedVolumeIn,
                        const volume::Volume& extraVolumeIn,
                        const real weight,
                        const grid::Grid& grid
                        ) override;

        //! Returns the number of elements needed to represent the functional
        //!
        //! Returns {numberOfH, 1,1}
        virtual ivec3 getFunctionalSize(const grid::Grid& grid) const override;

    private:

        void computeStructure(alsfvm::volume::Volume& outputVolume,
                              const alsfvm::volume::Volume& input);


        //! Helper function, computes the volume integral
        //!
        //! \note This must be called in order according to the ordering of h
        //! ie h=0 must be called first, then h=1, etc.
        void computeCube(alsfvm::memory::View<real>& output,
                         const alsfvm::memory::View<const real>& input,
                         int i, int j, int k, int h, int nx, int ny, int nz,
                         int ngx, int ngy, int ngz, int dimensions);

        const real p;
        const int numberOfH;
    };
} // namespace functional
} // namespace alsfvm
