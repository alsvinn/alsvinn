#pragma once
#include "alsfvm/io/Writer.hpp"
#include <netcdf.h>
#include "alsfvm/io/netcdf_utils.hpp"

namespace alsfvm { namespace io { 

    //! The netcdf writer writes to the netcdf format.
    //! This is the recommended writer to use
    class NetCDFWriter : public Writer {
    public:
        //! @param basefileName the base filename to use. Resulting filenames
        //!                     will be of the form
        //!                         basefileName_<timestep>.nc
        //!
        NetCDFWriter(const std::string& basefileName);

        virtual ~NetCDFWriter() {}

        //! Write the volume to file
        virtual void write(const volume::Volume& conservedVariables,
                           const volume::Volume& extraVariables,
                           const grid::Grid& grid,
                           const simulator::TimestepInformation& timestepInformation);
    protected:
        virtual void writeToFile(netcdf_raw_ptr file, const volume::Volume& conservedVariables,
                                 const volume::Volume& extraVariables,
                                 const grid::Grid& grid,
                                 const simulator::TimestepInformation& timestepInformation);
        virtual std::array<netcdf_raw_ptr, 3> createDimensions(netcdf_raw_ptr basegroup, const volume::Volume &volume);
        virtual void writeMemory(netcdf_raw_ptr baseGroup, netcdf_raw_ptr dataset,  const volume::Volume &volume, size_t memoryIndex);

        virtual void writeVolume(netcdf_raw_ptr baseGroup, const volume::Volume& volume, std::array<netcdf_raw_ptr, 3> dimensions);

        //! Creates or opens a dataset for the given volume and memory index
        //! \returns a  touple where the first member is the file/group id, and the second the dataset
        virtual std::pair<netcdf_raw_ptr, netcdf_raw_ptr> makeDataset(netcdf_raw_ptr baseGroup, const volume::Volume& volume,
                                           size_t memoryIndex, std::array<netcdf_raw_ptr, 3> dimensions);

        //! Creates the next filename and increments snapshot number
        std::string getFilename();

        size_t snapshotNumber{0};
        const std::string basefileName;
    };
} // namespace io
} // namespace alsfvm
