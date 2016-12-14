#pragma once
#include "alsfvm/io/Writer.hpp"
#include "hdf5.h"
#include "alsfvm/io/hdf5_utils.hpp"

namespace alsfvm {
namespace io {

///
/// \brief The HDF5Writer class writes output to HDF5 format
/// \note This class writes the output as VizSchema4, see
///       https://ice.txcorp.com/trac/vizschema/wiki/WikiStart
/// \note The output can easily be opened in Visit and Paraview (but need new
///       paraview version)
///
class HDF5Writer : public alsfvm::io::Writer
{
public:
    ///
    /// \brief HDF5Writer constructs a new HDF5Writer
    /// \param basefileName the basefilename to use (this could be eg.
    ///                     "some_simulation".
    /// \note Timestep information will be added to the filename, as well as
    ///       proper extension (.h5).
    ///
    HDF5Writer(const std::string& basefileName);

    virtual ~HDF5Writer() {}

    ///
    /// \brief write writes the data to disk
    /// \param conservedVariables the conservedVariables to write
    /// \param extraVariables the extra variables to write
    /// \param grid the grid currently used (includes whole domain, in case of mpi)
    /// \param timestepInformation
    ///
    virtual void write(const volume::Volume& conservedVariables,
                       const volume::Volume& extraVariables,
					   const grid::Grid& grid,
                       const simulator::TimestepInformation& timestepInformation);

protected:

	///
	/// Writes the grid to the file in VizSchema format
	/// \param object the object to write the grid to 
	/// \param grid the grid to use
    /// \note This function creates a new group in object named "grid"
	///
	void writeGrid(hid_t object, const grid::Grid& grid);

    void writeTimeGroup(hid_t object, const simulator::TimestepInformation& timestepInformation);

    ///
    /// \brief writeVolume takes each variable of the volume and writes it
    /// \param volume the volume to read from
    /// \param file the file to write to
    ///
    void writeVolume(const volume::Volume& volume, hid_t file, hid_t accessList=H5P_DEFAULT);



    ///
    /// \brief writeMemory writes a memory area to disk
    /// \param volume the volume to read from
    /// \param index the index of the memory area to read from
    /// \param name the name of the memory (variable name)
    /// \param file the file to write to
    ///
    void writeMemory(const volume::Volume& volume, size_t index, const std::string& name,
                     hid_t file, hid_t accessList=H5P_DEFAULT);

    ///
    /// \brief createDatasetForMemroy creates a dataset for the given memory
    /// \param volume the volume to read from
    /// \param index the index of the memory area to read from
    /// \param name the name of the memory (variable name)
    /// \param file the file to write to
    ///
    virtual std::unique_ptr<HDF5Resource> createDatasetForMemory(const volume::Volume& volume, size_t index, const std::string& name,
                     hid_t file);

    ///
    /// \brief createDatasetForMemroy creates a dataset for the given memory
    /// \param volume the volume to read from
    /// \param index the index of the memory area to read from
    /// \param name the name of the memory (variable name)
    /// \param file the file to write to
    ///
    void writeMemoryToDataset(const volume::Volume& volume, size_t index, const std::string& name,
                     hid_t dataset, hid_t accessList=H5P_DEFAULT);



    ///
    /// \brief writeString writes the string as an attribute to the given object
    /// \param object the id of the (opened) object to write to
    /// \param name the name of the attribute
    /// \param value the string value
    ///
    void writeString(hid_t object, const std::string& name, const std::string& value);


    ///
    /// \brief writeFloats writes the vector of floats as an attribute
    /// \param object the object to write to
    /// \param name the name of the attribute
    /// \param values the values to write
    ///
    void writeFloats(hid_t object, const std::string& name, const std::vector<float>& values);

    ///
    /// \brief writeIntegerss writes the vector of integers as an attribute
    /// \param object the object to write to
    /// \param name the name of the attribute
    /// \param values the values to write
    ///
    void writeIntegers(hid_t object, const std::string& name, const std::vector<int>& values);


    size_t snapshotNumber;
    const std::string basefileName;
};

} // namespace io
} // namespace alsfvm


