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
