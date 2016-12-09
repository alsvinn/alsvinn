#pragma once
#include "alsuq/generator/Generator.hpp"
namespace alsuq { namespace distribution { 

    class Distribution {
    public:
        virtual ~Distribution() {};
        virtual real generate(generator::Generator& generator, size_t component) = 0;
    };
} // namespace distribution
} // namespace alsuq
