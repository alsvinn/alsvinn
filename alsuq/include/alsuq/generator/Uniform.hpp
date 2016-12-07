#pragma once
#include "alsuq/generator/Generator.hpp"
#include "alsuq/generator/Parameters.hpp"
#include <random>

namespace alsuq { namespace generator { 

    class Uniform : public Generator {
    public:
        Uniform(const Parameters& parameters);

        real generate(size_t component);
    private:
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution;
    };
} // namespace generator
} // namespace alsuq
