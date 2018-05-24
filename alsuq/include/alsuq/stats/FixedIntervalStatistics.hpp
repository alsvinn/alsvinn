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
#include "alsuq/stats/Statistics.hpp"
#include "alsfvm/integrator/TimestepAdjuster.hpp"
#include "alsuq/types.hpp"
namespace alsuq {
namespace stats {

//! Decorator for the statistics class to only write a given interval, this
//! mimics the use of ::alsfvm::io::FixedIntervalWriter
//!
class FixedIntervalStatistics : public Statistics,
    public alsfvm::integrator::TimestepAdjuster {
public:

    ///
    /// \param writer the underlying writer to actually use.
    /// \param timeInterval the time interval (will save for every time n*timeInterval)
    /// \param endTime the final time for the simulation.
    ///
    FixedIntervalStatistics(alsfvm::shared_ptr<Statistics>& writer,
        real timeInterval, real endTime);


    virtual real adjustTimestep(real dt,
        const alsfvm::simulator::TimestepInformation& timestepInformation) const
    override;

    //! To be called when the statistics should be combined.
    virtual void combineStatistics() override;

    //! Adds a write for the given statistics name
    //! @param name the name of the statitics (one of the names returned in
    //!             getStatiticsNames()
    //! @param writer the writer to use
    virtual void addWriter(const std::string& name,
        std::shared_ptr<alsfvm::io::Writer>& writer) override;

    //! Returns a list of the names of the statistics being computed,
    //! typically this could be ['mean', 'variance']
    virtual std::vector<std::string> getStatisticsNames() const override;

    void writeStatistics(const alsfvm::grid::Grid& grid) override;


    //! To be called in the end, this could be to eg compute the variance
    //! through M_2-mean^2 or any other postprocessing needed
    virtual void finalizeStatistics() override;

protected:
    virtual void computeStatistics(const alsfvm::volume::Volume& conservedVariables,
        const alsfvm::volume::Volume& extraVariables,
        const alsfvm::grid::Grid& grid,
        const alsfvm::simulator::TimestepInformation& timestepInformation) override;

private:
    alsfvm::shared_ptr<Statistics> statistics;
    const real timeInterval;
    const real endTime;
    size_t numberSaved = 0;

};
} // namespace stats
} // namespace alsuq
