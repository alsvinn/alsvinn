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
