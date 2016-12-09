#pragma once
#include "alsuq/samples/SampleGenerator.hpp"
#include "alsuq/run/SimulatorCreator.hpp"
namespace alsuq { namespace run { 

    class Runner {
    public:
        Runner(std::shared_ptr<SimulatorCreator> simulatorCreator,
               std::shared_ptr<samples::SampleGenerator> sampleGenerator,
               std::vector<size_t> sampleNumbers);




        void run();

    private:
        std::shared_ptr<SimulatorCreator> simulatorCreator;
        std::shared_ptr<samples::SampleGenerator> sampleGenerator;
        std::vector<std::string> parameterNames;
        std::vector<size_t> sampleNumbers;
    };
} // namespace run
} // namespace alsuq
