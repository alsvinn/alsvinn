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

namespace alsuq {
namespace stats {

//! Simple timer class
class StatisticsTimer : public Statistics {
public:
    StatisticsTimer(const std::string& name,
        std::shared_ptr<Statistics> statistics);

    ~StatisticsTimer();

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


    virtual void computeStatistics(const alsfvm::volume::Volume& conservedVariables,
        const alsfvm::grid::Grid& grid,
        const alsfvm::simulator::TimestepInformation& timestepInformation) override;

    //! To be called in the end, this could be to eg compute the variance
    //! through M_2-mean^2 or any other postprocessing needed
    virtual void finalizeStatistics() override;

    virtual void writeStatistics(const alsfvm::grid::Grid& grids) override;

private:
    std::string name;
    const std::shared_ptr<Statistics> statistics;

    double statisticsTime = 0;
    double combineTime = 0;
    double finalizeTime = 0;
};
} // namespace stats
} // namespace alsuq
