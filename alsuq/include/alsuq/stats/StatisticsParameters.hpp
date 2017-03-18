#pragma once
#include "alsuq/types.hpp"
#include <boost/property_tree/ptree.hpp>
namespace alsuq { namespace stats {

    class StatisticsParameters {
    public:

        void setNumberOfSamples(size_t samples);
        size_t getNumberOfSamples() const;

        const std::string &getParameterAsString(const std::string& name) const;


        real getParameterAsDouble(const std::string& name) const;

        void setConfiguration(const boost::property_tree::ptree& configuration);
    private:
        size_t samples;
        boost::property_tree::ptree configuration;
    };
} // namespace stats
} // namespace alsuq
