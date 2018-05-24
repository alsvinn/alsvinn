#pragma once
#include "alsfvm/volume/Volume.hpp"
#include "alsfvm/grid/Grid.hpp"
#include "alsfvm/equation/CellComputer.hpp"
#include "alsfvm/init/Parameters.hpp"
#include <boost/property_tree/ptree.hpp>
namespace alsfvm {
namespace init {

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


    virtual void setParameters(const Parameters& parameters) = 0;

    //! Should provide a description of the initial data (eg the python script
    //! used for the initial data). Does not need to be machine parseable in any
    //! way, this is for "human readable reproducability" and extra debugging information.
    virtual boost::property_tree::ptree getDescription() const = 0;
};
} // namespace alsfvm
} // namespace init
