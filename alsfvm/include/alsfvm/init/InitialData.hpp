#pragma once
#include "alsfvm/volume/Volume.hpp"
#include "alsfvm/grid/Grid.hpp"
#include "alsfvm/equation/CellComputer.hpp"

namespace alsfvm { namespace init { 

    class InitialData {
    public:
        virtual ~InitialData() {}
        ///
        /// \brief setInitialData sets the initial data
        /// \param conservedVolume conserved volume to fill
        /// \param extraVolume the extra volume
        /// \param cellComputer an instance of the cell computer for the equation
        /// \param primitiveVolume an instance of the primtive volume for the equation
        /// \param grid underlying grid.
        /// \note All volumes need to have the correct size. All volumes will at the
        /// end be written to.
        /// \note This is not an efficient implementation, so it should really only
        /// be used for initial data!
        ///
        virtual void setInitialData(volume::Volume& conservedVolume,
                                    volume::Volume& extraVolume,
                                    volume::Volume& primitiveVolume,
                                    equation::CellComputer& cellComputer,
                                    grid::Grid& grid) = 0;

    };
} // namespace alsfvm
} // namespace init
