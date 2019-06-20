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
class CoarseGrainingIntervalWriter : public Writer,
    public integrator::TimestepAdjuster {
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
    /// \param grid the grid that is used (describes the _whole_ domain)
    /// \param timestepInformation
    ///
    virtual void write(const volume::Volume& conservedVariables,
        const grid::Grid& grid,
        const simulator::TimestepInformation& timestepInformation);

    virtual real adjustTimestep(real dt,
        const simulator::TimestepInformation& timestepInformation) const;

    virtual void finalize(const grid::Grid& grid,
        const simulator::TimestepInformation& timestepInformation) override;
private:
    alsfvm::shared_ptr<Writer> writer;
    const real timeInterval;
    const int numberOfCoarseSaves;
    const int numberOfSkips;

    int numberSaved{0};
    int numberSmallSaved{0};
    bool first{true};
    real dx;
};
} // namespace alsfvm
} // namespace io
