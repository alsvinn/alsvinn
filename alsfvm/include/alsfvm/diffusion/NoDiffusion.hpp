#pragma once
#include "alsfvm/diffusion/DiffusionOperator.hpp"
namespace alsfvm { namespace diffusion { 

    //! Null object for diffusion. Doesn't actually add any diffusion
    //! at all.
    class NoDiffusion : public DiffusionOperator {
    public:
        virtual void applyDiffusion(volume::Volume& outputVolume,
            const volume::Volume& conservedVolume) ;


        /// Gets the total number of ghost cells this diffusion needs,
        /// this is typically governed by reconstruction algorithm.
        virtual size_t getNumberOfGhostCells() const;
    };
} // namespace diffusion
} // namespace alsfvm
