#pragma once
#include <boost/property_tree/ptree.hpp>
#include "alsuq/samples/SampleGenerator.hpp"
#include "alsuq/mpi/Config.hpp"
#include "alsuq/run/Runner.hpp"
#include "alsuq/stats/Statistics.hpp"
namespace alsuq { namespace config { 

    class Setup {
    public:
        typedef boost::property_tree::ptree ptree;

        std::shared_ptr<run::Runner> makeRunner(const std::string& inputFilename,
                                                mpi::Config& mpiConfig);

    private:

        std::shared_ptr<samples::SampleGenerator> makeSampleGenerator(ptree& configuration);

        std::vector<std::shared_ptr<stats::Statistics> > createStatistics(ptree& configuration);
        size_t readNumberOfSamples(ptree& configuration);
    };
} // namespace config
} // namespace alsuq
