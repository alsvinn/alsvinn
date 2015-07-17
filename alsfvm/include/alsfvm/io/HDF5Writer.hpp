#pragma once
#include "alsfvm/io/Writer.hpp"
#include "hdf5.h"

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

    ///
    /// \brief write writes the data to disk
    /// \param conservedVariables the conservedVariables to write
    /// \param extraVariables the extra variables to write
    /// \param timestepInformation
    ///
    virtual void write(const volume::Volume& conservedVariables,
                       const volume::Volume& extraVariables,
                       const simulator::TimestepInformation& timestepInformation);

protected:
    ///
    /// \brief writeVolume takes each variable of the volume and writes it
    /// \param volume the volume to read from
    /// \param file the file to write to
    ///
    void writeVolume(const volume::Volume& volume, hid_t file);

    ///
    /// \brief writeMemory writes a memory area to disk
    /// \param memory the memory area to write
    /// \param name the name of the memory (variable name)
    /// \param file the file to write to
    ///
    void writeMemory(const memory::Memory<real>& memory, const std::string& name,
                     hid_t file);

private:
    size_t snapshotNumber;
    std::string basefileName;
};

} // namespace io
} // namespace alsfvm

