#pragma once
#include "alsuq/stats/StatisticsHelper.hpp"
#include "alsuq/stats/StatisticsParameters.hpp"

namespace alsuq {
namespace stats {

class OnePointMoment : public StatisticsHelper {
public:
    OnePointMoment(const StatisticsParameters& parameters);

    //! Returns 'm<n>' where n is the moment
    virtual std::vector<std::string> getStatisticsNames() const override;



    virtual void computeStatistics(const alsfvm::volume::Volume& conservedVariables,
        const alsfvm::volume::Volume& extraVariables,
        const alsfvm::grid::Grid& grid,
        const alsfvm::simulator::TimestepInformation& timestepInformation) override;

    virtual void finalizeStatistics() override;
private:
    const int p;
    const std::string statisticsName;
};
} // namespace stats
} // namespace alsuq
