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
class HDF5MPIWriter : public HDF5Writer
{
public:
    ///
    /// \brief HDF5MPIWriter constructs a new HDF5Writer
    /// \param basefileName the basefilename to use (this could be eg.
    ///                     "some_simulation".
    /// \param groupNames names of groups to create in the file
    ///        (this is especially useful for MPI)
    /// \param groupIndex the groupIndex to write to
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
    /// \param extraVariables the extra variables to write
    /// \param grid the grid that is used (describes the _whole_ domain)
    /// \param timestepInformation
    ///
    virtual void write(const volume::Volume& conservedVariables,
                       const volume::Volume& extraVariables,
                       const grid::Grid& grid,
                       const simulator::TimestepInformation& timestepInformation);

protected:
    ///
    /// \brief createDatasetForMemroy creates a dataset for the given memory
    /// \param volume the volume to read from
    /// \param index the index of the memory area to read from
    /// \param name the name of the memory (variable name)
    /// \param file the file to write to
    ///
    virtual std::unique_ptr<HDF5Resource> createDatasetForMemory(const volume::Volume& volume, size_t index, const std::string& name,
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
