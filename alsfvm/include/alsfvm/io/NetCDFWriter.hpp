#pragma once
#include "alsfvm/io/Writer.hpp"
#include <netcdf.h>
#include "alsfvm/io/netcdf_utils.hpp"

namespace alsfvm {
namespace io {

//! The netcdf writer writes to the netcdf format.
//! This is the recommended writer to use
//!
//! @note can easily be read by the netcdf python package, see
//!       http://www.hydro.washington.edu/~jhamman/hydro-logic/blog/2013/10/12/plot-netcdf-data/
class NetCDFWriter : public Writer {
public:
    //! Creates a new instance of the NetCDFWriter
    //!
    //! @param basefileName the base filename to use. Resulting filenames
    //!                     will be of the form
    //!                         basefileName_<timestep>.nc
    //!
    NetCDFWriter(const std::string& basefileName);

    //! Since we inherit from this class, this is the safest option
    //! (the destructor is anyway empty)
    virtual ~NetCDFWriter() {}

    //! Write the volume to file
    //! This will create a variable for each volume
    //!
    //! There will be no additioanl grid information written,
    //! this is implicit in the netcdf format
    virtual void write(const volume::Volume& conservedVariables,
        const volume::Volume& extraVariables,
        const grid::Grid& grid,
        const simulator::TimestepInformation& timestepInformation);
protected:
    //! Writes to the opened file
    //! \@note Assumes the file is in define mode
    //!
    //! @param file the filepointer
    //! @param conservedVariables the conservedVariables to write
    //! @param extraVariables the extraVariables to write
    //! @param grid the underlying grid
    //! @param timestepInformation the current timestep information
    void writeToFile(netcdf_raw_ptr file,
        const volume::Volume& conservedVariables,
        const volume::Volume& extraVariables,
        const grid::Grid& grid,
        const simulator::TimestepInformation& timestepInformation);

    //! Creates the dimensions
    //! \note Can only be called once per file!
    //!
    //! @param basegroup the file pointer to write to
    //! @param volume will use this to get the dimensions
    //!
    std::array<netcdf_raw_ptr, 3> createDimensions(netcdf_raw_ptr basegroup,
        const volume::Volume& volume);

    //! Writes the memory to the given memory dataset
    //!
    //! @param baseGroup the file pointer (or group pointer)
    //! @param dataset the dataset to write to
    //! @param volume the volume to extract the memory from (we need this to get the sizes)
    //! @param memoryIndex the memoryIndex of the volume
    void writeMemory(netcdf_raw_ptr baseGroup, netcdf_raw_ptr dataset,
        const volume::Volume& volume, size_t memoryIndex);


    //! Writes the volume to file
    //!
    //! @param baseGroup the baseGroup to write to
    //! @param volume the given volume
    //! @param dimensions already created dimensions
    void writeVolume(netcdf_raw_ptr baseGroup, const volume::Volume& volume,
        std::array<netcdf_raw_ptr, 3> dimensions);

    //! Creates or opens a dataset for the given volume and memory index
    //!
    //! @param baseGroup the file/group to create the dimensions in
    //! @param volume the volume to extract the size information from
    //! @param memoryIndex the given memoryIndex  (used for the name)
    //! @param dimensions the already created dimensions
    //! \returns a  touple where the first member is the file/group id, and the second the dataset
    std::pair<netcdf_raw_ptr, netcdf_raw_ptr> makeDataset(
        netcdf_raw_ptr baseGroup, const volume::Volume& volume,
        size_t memoryIndex, std::array<netcdf_raw_ptr, 3> dimensions);

    //! Creates the next filename and increments snapshot number
    //!
    //! @note should only be called once per write!
    std::string getFilename();

    //! Writes basic Alsvinn info to file.
    void addFileInformation(netcdf_raw_ptr file);

    size_t snapshotNumber{0};
    const std::string basefileName;
};
} // namespace io
} // namespace alsfvm
