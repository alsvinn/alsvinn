#pragma once
#include "alsuq/distribution/Distribution.hpp"
#include "alsuq/distribution/Parameters.hpp"
namespace alsuq { namespace distribution { 

    class Normal : public Distribution{
    public:
        Normal(const Parameters& parameters);

        real generate(generator::Generator& generator, size_t component);

    private:
        real scale(real x);
        const real mean;
        const real standardDeviation;

        real buffer{42};
        bool hasBuffer{false};

    };
} // namespace distribution
} // namespace alsuq
