#pragma once
#include "alsfvm/io/Writer.hpp"
#include "alsfvm/integrator/TimestepAdjuster.hpp"

#include <memory>

namespace alsfvm { namespace io { 

///
/// \brief The CoarseGrainingIntervalWriter class is a decorator for another writer.
/// Its purpose is to only call the underlying Writer object at fixed time intervals.
///
/// This class is useful if you only want to save every x seconds of simulation.
///
/// \note this is used specifically for the coarse graining algorithm, and will save
///       a number of timesteps around a "main" time. eg it will save
///
/// \f[ T-n\Delta x, T-(n-1)\Delta x,\ldots,T-\Delta x, T, T+\Delta x,\ldots, T+n\Delta x \f]
///
class CoarseGrainingIntervalWriter : public Writer, public integrator::TimestepAdjuster
{
public:
    ///
    /// \param writer the underlying writer to actually use.
    /// \param timeInterval the time interval (will save for every time n*timeInterval)
    /// \param numberOfCoarseSaves the number of saves around a time save (corresponds to \$f\$f in the explanation for the class)
    /// \param endTime the final time for the simulation.
    /// \param numberOfSkips the number of timesteps to skip
    ///
    CoarseGrainingIntervalWriter(alsfvm::shared_ptr<Writer>& writer,
                                 real timeInterval,
                                 int numberOfCoarseSaves,
                                 real endTime,
                                 int numberOfSkips);

    virtual ~CoarseGrainingIntervalWriter() {}
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
    const int numberOfCoarseSaves;
    const int numberOfSkips;
    const real endTime;
    int numberSaved{0};
    int numberSmallSaved{0};
    bool first{true};
    real dx;
};
} // namespace alsfvm
} // namespace io
