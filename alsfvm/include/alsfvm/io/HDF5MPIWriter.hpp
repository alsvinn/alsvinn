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
#include "alsfvm/io/HDF5Writer.hpp"
#include <mpi.h>

namespace alsfvm {
namespace io {

///
/// \brief The HDF5MPIWriter write to the HDF5 format with MPI support
///
class HDF5MPIWriter : public HDF5Writer {
public:
    ///
    /// \brief HDF5MPIWriter constructs a new HDF5Writer
    /// \param basefileName the basefilename to use (this could be eg.
    ///                     "some_simulation".
    /// \param groupNames names of groups to create in the file
    ///        (this is especially useful for MPI)
    /// \param groupIndex the groupIndex to write to
    ///
    /// \param mpiCommunicator the given mpiCommunicator (used for pNETCDF)
    ///
    /// \param newFile creates a new file if true, otherwise tries to open
    ///                an already existing file (will fail if it does not exist
    ///                or does not agree with our format). When in doubt,
    ///                set it true for first sample save, then false
    ///
    /// \param mpiInfo the mpiInfo (passed to pNetCDF)
    ///
    ///
    /// \note Timestep information will be added to the filename, as well as
    ///       proper extension (.h5).
    ///
    HDF5MPIWriter(const std::string& basefileName,
        const std::vector<std::string>& groupNames,
        size_t groupIndex,
        bool newFile,
        MPI_Comm mpiCommunicator,
        MPI_Info mpiInfo);

    // We will inherit from this, hence virtual destructor.
    virtual ~HDF5MPIWriter() {}


    ///
    /// \brief write writes the data to disk
    /// \param conservedVariables the conservedVariables to write
    /// \param grid the grid that is used (describes the _whole_ domain)
    /// \param timestepInformation
    ///
    virtual void write(const volume::Volume& conservedVariables,
        const grid::Grid& grid,
        const simulator::TimestepInformation& timestepInformation) override;

protected:
    ///
    /// \brief createDatasetForMemroy creates a dataset for the given memory
    /// \param volume the volume to read from
    /// \param index the index of the memory area to read from
    /// \param name the name of the memory (variable name)
    /// \param file the file to write to
    ///
    virtual std::unique_ptr<HDF5Resource> createDatasetForMemory(
        const volume::Volume& volume, size_t index, const std::string& name,
        hid_t file);
private:

    const std::vector<std::string> groupNames;
    const size_t groupIndex;
    const bool newFile;
    MPI_Comm mpiCommunicator;
    MPI_Info mpiInfo;

};

} // namespace io
} // namespace alsfvm
