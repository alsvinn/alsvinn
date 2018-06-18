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
#include "alsuq/types.hpp"
namespace alsuq {
namespace stats {

//! Computes the sturcture function given a direction
//!
//! Computes the structure function as
//!
//! \f[\sum_{i,j,k} \sum_{\tilde{i}=i-h}^{i+h}\cdot \sum_{\tilde{k}=k-h}^{k+h}|u_{(\tilde{i},\tilde{j},\tilde{i})}-u_{(i,j,k)}|^p/h^2\f]
class StructureCube : public StatisticsHelper {
public:
    StructureCube(const StatisticsParameters& parameters);


    //! Returns a list of ['structure_basic']
    virtual std::vector<std::string> getStatisticsNames() const override;




    virtual void computeStatistics(const alsfvm::volume::Volume& conservedVariables,
        const alsfvm::volume::Volume& extraVariables,
        const alsfvm::grid::Grid& grid,
        const alsfvm::simulator::TimestepInformation& timestepInformation) override;

    virtual void finalizeStatistics() override;



private:
    void computeStructure(alsfvm::volume::Volume& outputVolume,
        const alsfvm::volume::Volume& input);


    //! Helper function, computes the volume integral
    //!
    //! \note This must be called in order according to the ordering of h
    //! ie h=0 must be called first, then h=1, etc.
    void computeCube(alsfvm::memory::View<real>& output,
        const alsfvm::memory::View<const real>& input,
        int i, int j, int k, int h, int nx, int ny, int nz,
        int ngx, int ngy, int ngz, int dimensions);

    const real p;
    const int numberOfH;
    const std::string statisticsName;

};
} // namespace stats
} // namespace alsuq
