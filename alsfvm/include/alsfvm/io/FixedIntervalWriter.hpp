#pragma once
#include "alsfvm/io/Writer.hpp"

#include <memory>

namespace alsfvm { namespace io { 

///
/// \brief The FixedIntervalWriter class is a decorator for another writer.
/// Its purpose is to only call the underlying Writer object at fixed time intervals.
///
/// This class is useful if you only want to save every x seconds of simulation.
///
class FixedIntervalWriter : public Writer
{
public:
    ///
    /// \param writer the underlying writer to actually use.
    /// \param timeInterval the time interval (will save for every time n*timeInterval)
    /// \param endTime the final time for the simulation.
    ///
    FixedIntervalWriter(boost::shared_ptr<Writer>& writer, real timeInterval, real endTime);

    virtual ~FixedIntervalWriter() {}
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

private:
    boost::shared_ptr<Writer> writer;
    const real timeInterval;
    const real endTime;
    size_t numberSaved;

};
} // namespace alsfvm
} // namespace io
