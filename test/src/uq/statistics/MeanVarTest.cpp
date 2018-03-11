#include <gtest/gtest.h>
#include "alsuq/stats/StatisticsFactory.hpp"
using namespace alsuq::stats;
TEST(MeanVarStatistics, ConstructTest) {
    StatisticsParameters params{boost::property_tree::ptree()};
    StatisticsFactory factory;
    auto meanVar = factory.makeStatistics("cpu", "meanvar", params);



    ASSERT_EQ(2, meanVar->getStatisticsNames().size());
    ASSERT_EQ("mean", meanVar->getStatisticsNames()[0]);
    ASSERT_EQ("variance", meanVar->getStatisticsNames()[1]);
}
