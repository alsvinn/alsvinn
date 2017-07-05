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
