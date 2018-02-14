#pragma once
#include "alsuq/stats/StatisticsHelper.hpp"
#include "alsuq/types.hpp"
namespace alsuq {
namespace stats {

//! Computes the sturcture function given a direction
//!
//!
//! It then computes the structure function as
//!
//! \f[\sum_{i,j,k} \sum_{n,m,o\in N(i,j,k,h)}|u_{(n,m,o)}-u_{(i,j,k)}|^p\f]
class StructureSurface : public StatisticsHelper {
public:
    StructureSurface(const StatisticsParameters& parameters);


    //! Returns a list of ['structure_basic']
    virtual std::vector<std::string> getStatisticsNames() const;




    virtual void computeStatistics(const alsfvm::volume::Volume& conservedVariables,
        const alsfvm::volume::Volume& extraVariables,
        const alsfvm::grid::Grid& grid,
        const alsfvm::simulator::TimestepInformation& timestepInformation) override;

    virtual void finalize() override;



private:
    void computeStructure(alsfvm::volume::Volume& outputVolume,
        const alsfvm::volume::Volume& input);


    const real p;

    const size_t numberOfH;
    const std::string statisticsName;

};
} // namespace stats
} // namespace alsuq
