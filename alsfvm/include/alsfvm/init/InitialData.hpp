#pragma once
#include "alsfvm/volume/Volume.hpp"
#include "alsfvm/grid/Grid.hpp"

namespace alsfvm { namespace init { 

    class InitialData {
    public:

        ///
        /// \brief setInitialData sets the initial data
        /// \param conservedVolume conserved volume to fill
        /// \param extraVolume the extra volume
        /// \param grid underlying grid.
        ///
        virtual void setInitialData(volume::Volume& conservedVolume,
                            volume::Volume& extraVolume,
                            grid::Grid& grid) = 0;
    };
} // namespace alsfvm
} // namespace init
