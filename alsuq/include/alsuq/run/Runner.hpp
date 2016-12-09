#pragma once
#include "alsuq/samples/SampleGenerator.hpp"
namespace alsuq { namespace run { 

    class Runner {
    public:

        Runner(std::shared_ptr<samples::SampleGenerator> sampleGenerator);



    private:
        samples::SampleGenerator sampleGenerator;

    };
} // namespace run
} // namespace alsuq
