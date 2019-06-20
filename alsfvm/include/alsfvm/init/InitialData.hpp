/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
    /// \param cellComputer an instance of the cell computer for the equation
    /// \param primitiveVolume an instance of the primtive volume for the equation
    /// \param grid underlying grid.
    /// \note All volumes need to have the correct size. All volumes will at the
    /// end be written to.
    /// \note This is not an efficient implementation, so it should really only
    /// be used for initial data!
    ///
    virtual void setInitialData(volume::Volume& conservedVolume,
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
