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

//! Decorator to compute time averaged statistics. This will roughly work in the following way:
//!
//! It will call the underlying statistics class for each time tau where |tau-time|<timeRadius.
//!
class TimeIntegratedWriter : public Statistics {
public:

    ///
    /// \param writer the underlying writer to actually use.
    /// \param time the time for which to write the statistics
    /// \param timeRadius the radius of the time interval
    ///
    TimeIntegratedWriter(alsfvm::shared_ptr<Statistics>& writer, real time,
        real timeRadius);
    \

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
        const alsfvm::grid::Grid& grid,
        const alsfvm::simulator::TimestepInformation& timestepInformation) override;

private:
    alsfvm::shared_ptr<Statistics> statistics;

};
} // namespace stats
} // namespace alsuq
