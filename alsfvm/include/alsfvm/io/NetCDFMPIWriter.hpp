#pragma once
#include <mpi.h>
#include "alsfvm/io/NetCDFWriter.hpp"
namespace alsfvm { namespace io { 

    //! Writes to the mpi version of netcdf.
    //! \note Due to the new mpi of pNetCDF, this can not be combined
    //! in any meaningful way with the old NetCDFWriter class, the code is pretty
    //! much disjoint.
    class NetCDFMPIWriter : public NetCDFWriter {
    public:
        ///
        /// \brief NetCDFMPIWriter constructs a new NetCDFMPIWriter
        /// \param basefileName the basefilename to use (this could be eg.
        ///                     "some_simulation".
        /// \param groupNames names of groups to create in the file
        ///        (this is especially useful for MPI)
        /// \param groupIndex the groupIndex to write to
        ///
        /// \note Timestep information will be added to the filename, as well as
        ///       proper extension (.h5).
        ///
        NetCDFMPIWriter(const std::string& basefileName,
                      const std::vector<std::string>& groupNames,
                      size_t groupIndex,
                      bool newFile,
                      MPI_Comm mpiCommunicator,
                      MPI_Info mpiInfo);

        // We will inherit from this, hence virtual destructor.
        virtual ~NetCDFMPIWriter() {}


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
        typedef std::array<netcdf_raw_ptr, 3> dimension_vector;

        virtual dimension_vector createDimensions(netcdf_raw_ptr basegroup, const volume::Volume &volume,
                                                  bool newFile);


        //! Creates or opens a dataset for the given volume and memory index
        //! \returns a  touple where the first member is the file/group id, and the second the dataset
        virtual std::vector<netcdf_raw_ptr>
        makeDataset(netcdf_raw_ptr baseGroup, const volume::Volume& volume,
                    std::array<netcdf_raw_ptr, 3> dimensions);


        virtual void writeToFile(netcdf_raw_ptr file, const volume::Volume& conservedVariables,
                                 const volume::Volume& extraVariables,
                                 const grid::Grid& grid,
                                 const simulator::TimestepInformation& timestepInformation,
                                 bool newFile);

        virtual void writeMemory(netcdf_raw_ptr baseGroup, netcdf_raw_ptr dataset,  const volume::Volume &volume, size_t memoryIndex);

        virtual void writeVolume(netcdf_raw_ptr baseGroup, const volume::Volume& volume, std::array<netcdf_raw_ptr, 3> dimensions,
                                 const std::vector<netcdf_raw_ptr>& datasets);


    private:
        const std::vector<std::string> groupNames;
        const size_t groupIndex;
        const bool newFile;
        MPI_Comm mpiCommunicator;
        MPI_Info mpiInfo;

    };
} // namespace io
} // namespace alsfvm
