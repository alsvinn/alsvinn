#pragma once
#include <mpi.h>
#include "alsfvm/io/NetCDFWriter.hpp"
namespace alsfvm { namespace io { 

    //! Writes to the mpi version of netcdf.
    //! @note Due to the new mpi of pNetCDF, this can not be combined
    //! in any meaningful way with the old NetCDFWriter class, the code is pretty
    //! much disjoint.
    //!
    //! @note can easily be read by the netcdf python package, see
    //!       http://www.hydro.washington.edu/~jhamman/hydro-logic/blog/2013/10/12/plot-netcdf-data/
    class NetCDFMPIWriter : public NetCDFWriter {
    public:
        ///
        /// \brief NetCDFMPIWriter constructs a new NetCDFMPIWriter
        /// \param basefileName the basefilename to use (this could be eg.
        ///                     "some_simulation".
        /// \param groupNames names of groups to create in the file
        ///        (this is especially useful for MPI). If left blank (""), no prefix will be given
        /// \param groupIndex the groupIndex to write to
        ///
        /// \param newFile should we create (or overwrite) the file? If false,
        ///                the file will be opened and fail if it does not exist
        ///                or if the data does not match (ie. if the sizes mismatch,
        ///                or if the datasets are named differently)
        ///
        /// \param mpiCommunicator the given mpiCommunicator (used for pNETCDF)
        ///
        /// \param mpiInfo the mpiInfo (passed to pNetCDF)
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

        //! We could inherit from this, hence virtual destructor.
        virtual ~NetCDFMPIWriter() {}


        ///
        /// \brief write writes the data to disk
        ///
        /// This writes the data in the format
        ///
        /// \code
        ///     <groupName>_<variable_name>
        /// \endcode
        ///
        /// we do not use any groups to store the file at the moment,
        /// this is to ensure maximal compatability with pNetCDF.
        ///
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

        //! Creates dimensions
        //!
        //! @note should only be called once per file
        //!
        //! @param basegroup the basegroup to use (or file)
        //! @param grid the grid from which to extract  size information
        //! @param newFile if true, we will create the dimensions, otherwise
        //!                we will try to open the dimensions from the file
        //!
        //! @return the dimensions created by netcdf
        virtual dimension_vector createDimensions(netcdf_raw_ptr basegroup, const grid::Grid &grid,
                                                  bool newFile);


        //! Creates or opens a dataset for the given volume
        //!
        //! This will create variables for all variables in the volume
        //!
        //! @note these should be called before writing data.
        //!
        //! @param baseGroup the file/group to write to
        //!
        //! @param volume volume is used for size and name information
        //!
        //! @param dimensions the already created dimensions
        //!
        //! @returns a list of pairs  where the first member is the file/group id,
        //!          and the second the dataset. It is ordered according to the volume
        virtual std::vector<netcdf_raw_ptr>
        makeDataset(netcdf_raw_ptr baseGroup, const volume::Volume& volume,
                    std::array<netcdf_raw_ptr, 3> dimensions);

        //! Writes to the opened file
        //! \@note Assumes the file is in define mode
        //!
        //! @param file the filepointer
        //! @param conservedVariables the conservedVariables to write
        //! @param extraVariables the extraVariables to write
        //! @param grid the underlying grid
        //! @param timestepInformation the current timestep information
        //! @param newFile is true if the file is created for this iteration, otherwise false
        virtual void writeToFile(netcdf_raw_ptr file, const volume::Volume& conservedVariables,
                                 const volume::Volume& extraVariables,
                                 const grid::Grid& grid,
                                 const simulator::TimestepInformation& timestepInformation,
                                 bool newFile);


        //! Writes the given memory to the dataset/variable
        //!
        //! @param baseGroup the basegroup/file to write to
        //! @param dataset the dataset to write to
        //! @param volume the volume is used to get size information
        //! @param memoryIndex the scalar memory index of the volume
        //!
        virtual void writeMemory(netcdf_raw_ptr baseGroup,
                                 netcdf_raw_ptr dataset,
                                 const volume::Volume &volume,
                                 size_t memoryIndex,
                                 const grid::Grid& grid);


        //! Writes the volume (ie looops over all memory areas and writes each memory area)
        //!
        //! @param baseGroup the baseGroup to use
        //! @param volume the volume to use
        //! @param dimensions the given dimensions
        //! @param datasets a list of datasets (produced by makeDataset) that  is
        //!                 ordered according to the volume (makeDataset does this automatically)
        //!
        //!
        virtual void writeVolume(netcdf_raw_ptr baseGroup,
                                 const volume::Volume& volume,
                                 std::array<netcdf_raw_ptr, 3> dimensions,
                                 const std::vector<netcdf_raw_ptr>& datasets,
                                 const grid::Grid& grid);


    private:
        const std::vector<std::string> groupNames;
        const size_t groupIndex;
        const bool newFile;
        MPI_Comm mpiCommunicator;
        MPI_Info mpiInfo;

    };
} // namespace io
} // namespace alsfvm
