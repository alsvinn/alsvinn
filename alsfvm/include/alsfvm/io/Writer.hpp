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
#include "alsfvm/simulator/TimestepInformation.hpp"
#include "alsfvm/volume/Volume.hpp"
#include "alsfvm/grid/Grid.hpp"
#include <boost/property_tree/ptree.hpp>

namespace alsfvm {
namespace io {

///
/// \brief The Writer class is an abstract interface to represent output writers
///
class Writer {
public:
    // We will inherit from this, hence virtual destructor.
    virtual ~Writer() {}


    ///
    /// \brief write writes the data to disk
    /// \param conservedVariables the conservedVariables to write
    /// \param extraVariables the extra variables to write
    /// \param grid the grid that is used (describes the _whole_ domain)
    /// \param timestepInformation
    ///
    virtual void write(const volume::Volume& conservedVariables,
        const volume::Volume& extraVariables,
        const grid::Grid& grid,
        const simulator::TimestepInformation& timestepInformation) = 0;


    //! This method should be called at the end of the simulation
    virtual void finalize(const grid::Grid& grid,
        const simulator::TimestepInformation& timestepInformation) {}

    //! Adds attributes to be written to the file (this is an optional
    //! feature, not every writer supports this. Attributes should be
    //! description of the simulation environment to help reproduce the output
    //! file (eg. numerical parameters, initial data, etc).
    void addAttributes(const std::string& nameOfAttributes,
        const boost::property_tree::ptree& attributes);

protected:
    std::map<std::string, boost::property_tree::ptree> attributesMap;

};

typedef alsfvm::shared_ptr<Writer> WriterPointer;

} // namespace io
} // namespace alsfvm
