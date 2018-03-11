#pragma once
#include "alsuq/stats/StatisticsHelper.hpp"
#include "alsfvm/functional/Functional.hpp"
namespace alsuq {
namespace stats {

//! This class takes any functional object and computes the
//! time integrated mean and variance of said functional
//!
//! parameters accepted are:
//!
//! parameter name               | description
//! -----------------------------|-----------------
//!  functional_name             | name of functional
//!  time                        | time point to integrated around
//!  timeRadius                  | the time radius
//!  other arguments           | gets passed to the functional directly
class TimeIntegratedFunctionalStatistics : public StatisticsHelper {
public:

    TimeIntegratedFunctionalStatistics(const StatisticsParameters& parameters);


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
    double lastTime = 0;
    double time = 0;
    double timeRadius = 0;

    alsfvm::simulator::TimestepInformation fixedTimestepInformation;
};
} // namespace stats
} // namespace alsuq
