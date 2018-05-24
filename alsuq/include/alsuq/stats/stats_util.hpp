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
#include "alsuq/stats/StatisticsFactory.hpp"
#include <boost/preprocessor.hpp>
#define REGISTER_STATISTICS(platform, name, classname) namespace { \
    static struct BOOST_PP_CAT(RegisterStruct, BOOST_PP_CAT(platform, name)) { \
        BOOST_PP_CAT(RegisterStruct, BOOST_PP_CAT(platform, name))() { \
            alsuq::stats::StatisticsFactory::registerStatistics(#platform, #name, [](const alsuq::stats::StatisticsParameters& params) { \
                std::shared_ptr<alsuq::stats::Statistics> statistics; \
                statistics.reset(new classname(params)); \
                return statistics; \
                });\
            } \
        } BOOST_PP_CAT(BOOST_PP_CAT(registerObject, platform),name); \
    }
