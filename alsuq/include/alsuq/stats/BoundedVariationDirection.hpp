#pragma once
#include "alsuq/stats/StatisticsHelper.hpp"

namespace alsuq { namespace stats {

    class BoundedVariationDirection : public StatisticsHelper {
    public:

        BoundedVariationDirection(const StatisticsParameters& parameters);


        //! Returns a list of ['bv_x', 'bv_y', 'bv_z']
        virtual std::vector<std::string> getStatisticsNames() const;




        virtual void computeStatistics(const alsfvm::volume::Volume &conservedVariables,
                          const alsfvm::volume::Volume &extraVariables,
                          const alsfvm::grid::Grid &grid,
                          const alsfvm::simulator::TimestepInformation &timestepInformation) override;

        virtual void finalize() override;



    private:


        const std::vector<std::string> statisticsNames = {"bv_x",
                                                          "bv_y",
                                                          "bv_z"};



    };
} // namespace statistics
} // namespace alsuq
