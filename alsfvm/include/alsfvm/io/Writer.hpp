#pragma once
#include "alsfvm/simulator/TimestepInformation.hpp"
#include "alsfvm/volume/Volume.hpp"

namespace alsfvm {
namespace io {

///
/// \brief The Writer class is an abstract interface to represent output writers
///
class Writer
{
public:
    // We will inherit from this, hence virtual destructor.
    virtual ~Writer() {}


    ///
    /// \brief write writes the data to disk
    /// \param conservedVariables the conservedVariables to write
    /// \param extraVariables the extra variables to write
    /// \param timestepInformation
    ///
    virtual void write(const volume::Volume& conservedVariables,
                       const volume::Volume& extraVariables,
                       const simulator::TimestepInformation& timestepInformation) = 0;

};

} // namespace io
} // namespace alsfvm
