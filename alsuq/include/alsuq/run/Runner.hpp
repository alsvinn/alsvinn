#pragma once
#include "alsuq/samples/SampleGenerator.hpp"
#include "alsuq/run/SimulatorCreator.hpp"
#include "alsuq/stats/Statistics.hpp"

namespace alsuq { namespace run { 

    class Runner {
    public:
        Runner(std::shared_ptr<SimulatorCreator> simulatorCreator,
               std::shared_ptr<samples::SampleGenerator> sampleGenerator,
               std::vector<size_t> sampleNumbers,
               mpi::ConfigurationPtr mpiConfig,
               const std::string& name);




        void run();


        //! Sets the statistics to be used
        void setStatistics(const std::vector<std::shared_ptr<stats::Statistics> >& statistics);
        std::string getName() const;

        size_t getTimestepsPerformedTotal() const;

    private:
        std::shared_ptr<SimulatorCreator> simulatorCreator;
        std::shared_ptr<samples::SampleGenerator> sampleGenerator;
        std::vector<std::string> parameterNames;
        std::vector<size_t> sampleNumbers;
        std::vector<std::shared_ptr<stats::Statistics> > statistics;

        mpi::ConfigurationPtr mpiConfig;
        const std::string name;
        size_t timestepsPerformedTotal = 0;
    };
} // namespace run
} // namespace alsuq
