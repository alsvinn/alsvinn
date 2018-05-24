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
#include "alsuq/stats/StatisticsHelper.hpp"
#include "alsfvm/functional/Functional.hpp"
namespace alsuq {
namespace stats {

//! This class takes any functional object and computes the
//! mean of said functional
//!
//! parameters accepted are:
//!
//!    parameter name            | description
//! -----------------------------|-----------------
//!  functionalName              | name of functional
//!  other arguments             | gets passed to the functional directly
class FunctionalStatistics : public StatisticsHelper {
public:

    FunctionalStatistics(const StatisticsParameters& parameters);


    //! Returns a list of ['mean_<functional_name>']
    virtual std::vector<std::string> getStatisticsNames() const override;




    virtual void computeStatistics(const alsfvm::volume::Volume& conservedVariables,
        const alsfvm::volume::Volume& extraVariables,
        const alsfvm::grid::Grid& grid,
        const alsfvm::simulator::TimestepInformation& timestepInformation) override;

    virtual void finalizeStatistics() override;


private:
    alsfvm::functional::FunctionalPointer functional = nullptr;
    std::vector<std::string> statisticsNames;

    std::string platform = "cpu";
};
} // namespace stats
} // namespace alsuq
