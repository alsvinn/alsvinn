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
