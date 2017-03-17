#pragma once
#include "alsuq/stats/StatisticsHelper.hpp"
#include "alsuq/stats/StatisticsParameters.hpp"

namespace alsuq { namespace stats { 

    class MeanVariance : public StatisticsHelper{
    public:
        MeanVariance(const StatisticsParameters& parameters);
        //! Returns a list of ['mean', 'variance']
        virtual std::vector<std::string> getStatisticsNames() const;
    protected:


        virtual void computeStatistics(const alsfvm::volume::Volume &conservedVariables,
                          const alsfvm::volume::Volume &extraVariables,
                          const alsfvm::grid::Grid &grid,
                          const alsfvm::simulator::TimestepInformation &timestepInformation) override;
    };
} // namespace stats
} // namespace alsuq
