#pragma once
#include "alsuq/distribution/Distribution.hpp"
#include "alsuq/distribution/Parameters.hpp"

namespace alsuq { namespace distribution { 

    class Uniform : public Distribution {
    public:
        Uniform(const Parameters& parameters);

        real generate(generator::Generator& generator, size_t component);

    private:
        real scale(real x);
        const real a;
        const real b;

    };
} // namespace distribution
} // namespace alsuq
