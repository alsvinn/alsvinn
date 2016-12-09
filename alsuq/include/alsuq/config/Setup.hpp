#pragma once
#include <boost/property_tree/ptree.hpp>
#include "alsuq/samples/SampleGenerator.hpp"
namespace alsuq { namespace config { 

    class Setup {
    public:
        typedef boost::property_tree::ptree ptree;
        std::shared_ptr<samples::SampleGenerator> makeSampleGenerator(ptree& configuration);
        size_t readNumberOfSamples(ptree& configuration);
    };
} // namespace config
} // namespace alsuq
