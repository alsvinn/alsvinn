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

#include "alsuq/stats/StatisticsFactory.hpp"
#include "alsutils/error/Exception.hpp"
#include "alsuq/stats/StatisticsTimer.hpp"

namespace alsuq {
namespace stats {

namespace {
class StatististicsList {


public:
    static StatististicsList& instance() {
        static StatististicsList list;

        return list;
    }
    std::map<std::string, std::map<std::string, StatisticsFactory::StatisticsCreator> >
    creators;
private:
    StatististicsList() {}

};
}

void StatisticsFactory::registerStatistics(const std::string& platform,
    const std::string& name,
    StatisticsFactory::StatisticsCreator maker) {
    auto& list = StatististicsList::instance().creators;

    if (list[platform].find(name) != list[platform].end()) {
        THROW("'" << name << "' already registered as a Statistic");
    }

    list[platform][name] = maker;
}

StatisticsFactory::StatisticsPointer StatisticsFactory::makeStatistics(
    const std::string& platform,
    const std::string& name,
    const StatisticsParameters& params) {
    auto& list = StatististicsList::instance().creators;

    if (list[platform].find(name) == list[platform].end()) {
        THROW("Unknown statistics: " << name);
    }


    StatisticsPointer pointer;
    pointer.reset(new StatisticsTimer(name, list[platform][name](params)));

    return pointer;
}

}
}
