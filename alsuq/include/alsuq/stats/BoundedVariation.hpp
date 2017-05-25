#pragma once
#include "alsuq/stats/StatisticsHelper.hpp"

namespace alsuq { namespace stats {

    class BoundedVariation : public StatisticsHelper {
    public:

        BoundedVariation(const StatisticsParameters& parameters);


        //! Returns a list of ['bv']
        virtual std::vector<std::string> getStatisticsNames() const;




        virtual void computeStatistics(const alsfvm::volume::Volume &conservedVariables,
                          const alsfvm::volume::Volume &extraVariables,
                          const alsfvm::grid::Grid &grid,
                          const alsfvm::simulator::TimestepInformation &timestepInformation) override;

        virtual void finalize() override;



    private:


        const std::string statisticsName = "bv";



    };
} // namespace statistics
} // namespace alsuq
