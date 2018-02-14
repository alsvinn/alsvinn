#pragma once
#include "alsuq/stats/Statistics.hpp"
#include "alsuq/stats/StatisticsParameters.hpp"
#include "alsuq/types.hpp"
#include <memory>
#include <functional>

namespace alsuq {
namespace stats {

class StatisticsFactory {
public:
    typedef std::shared_ptr<Statistics> StatisticsPointer;
    typedef std::function<StatisticsPointer(const StatisticsParameters&)>
    StatisticsCreator;
    static void registerStatistics(const std::string& platform,
        const std::string& name,
        StatisticsCreator maker);

    StatisticsPointer makeStatistics(const std::string& platform,
        const std::string& name,
        const StatisticsParameters& params);


};
} // namespace stats
} // namespace alsuq
