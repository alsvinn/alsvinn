#pragma once
#include "alsfvm/io/Writer.hpp"
#include "alsfvm/integrator/TimestepAdjuster.hpp"

#include <memory>

namespace alsfvm {
namespace io {

///
/// \brief The TimeIntegratedWriter class is a decorator for another writer.
/// Its purpose is to only call the underlying Writer object at fixed times around a time
///
/// This will save every timestep at time tau for which |tau-T|<delta, for
/// user specified T and delta.
///
class TimeIntegratedWriter : public Writer {
public:
    ///
    /// \param writer the underlying writer to actually use.
    /// \param time the center of the time to write to
    /// \param timeRadius the radius of the time ball to dump
    ///
    TimeIntegratedWriter(alsfvm::shared_ptr<Writer>& writer, real time,
        real timeRadius);

    virtual ~TimeIntegratedWriter() {}
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
    alsfvm::shared_ptr<Writer> writer;
    const real time;
    const real timeRadius;

    real lastTime = 0;
    bool written = false;
    alsfvm::shared_ptr<volume::Volume> integratedConservedVariables;
    alsfvm::shared_ptr<volume::Volume> integratedExtraVariables;


};
} // namespace alsfvm
} // namespace io
