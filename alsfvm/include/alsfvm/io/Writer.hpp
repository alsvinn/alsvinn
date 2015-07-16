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
    /// \param volume the volume to write to disk
    /// \param timestepInformation
    ///
    virtual void write(const volume::Volume& volume,
                       const simulator::TimestepInformation& timestepInformation) = 0;

};

} // namespace io
} // namespace alsfvm
