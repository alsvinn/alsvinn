/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
