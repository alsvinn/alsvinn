#pragma once
#include "alsfvm/volume/Volume.hpp"

namespace alsfvm { namespace diffusion { 

    /// Applies numerical diffusion to the given conserved variables
    ///
    /// This is typically used for the TeCNO-scheme, see
    /// http://www.cscamm.umd.edu/tadmor/pub/TV+entropy/Fjordholm_Mishra_Tadmor_SINUM2012.pdf
    class DiffusionOperator {
    public:
        virtual ~DiffusionOperator() {};

        /// Applies numerical diffusion to the outputVolume given the data in conservedVolume.
        ///
        /// \note The numerical diffusion will be added to outputVolume, ie. the code will 
        /// essentially work like
        /// \code{.cpp}
        /// outputVolume += diffusion(conservedVolume);
        /// \endcode
        virtual void applyDiffusion(volume::Volume& outputVolume, 
            const volume::Volume& conservedVolume) = 0;

    };
} // namespace alsfvm
} // namespace diffusion
