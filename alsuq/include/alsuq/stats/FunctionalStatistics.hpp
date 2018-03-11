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
