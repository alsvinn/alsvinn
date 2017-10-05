#pragma once
#include "alsfvm/io/Writer.hpp"
#include "alsfvm/integrator/TimestepAdjuster.hpp"

#include <memory>

namespace alsfvm { namespace io {

///
/// \brief The FixedIntervalWriter class is a decorator for another writer.
/// Its purpose is to only call the underlying Writer object at fixed time intervals.
///
/// This class is useful if you only want to save every x seconds of simulation.
///
class FixedIntervalWriter : public Writer, public integrator::TimestepAdjuster
{
public:
    ///
    /// \param writer the underlying writer to actually use.
    /// \param timeInterval the time interval (will save for every time n*timeInterval)
    /// \param endTime the final time for the simulation.
    ///
    FixedIntervalWriter(alsfvm::shared_ptr<Writer>& writer, real timeInterval, real endTime);

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

    virtual real adjustTimestep(real dt, const simulator::TimestepInformation &timestepInformation) const;

private:
    alsfvm::shared_ptr<Writer> writer;
    const real timeInterval;
    const real endTime;
    size_t numberSaved;

};
} // namespace alsfvm
} // namespace io
