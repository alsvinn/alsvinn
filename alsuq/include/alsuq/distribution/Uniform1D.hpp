#pragma once
#include "alsuq/distribution/Distribution.hpp"
namespace alsuq { namespace distribution { 

	//! Uses the midpoint rule of integration
	class Uniform1D : public Distribution {
    public:
	Uniform1D(size_t numberOfSamples, real a, real b);
        //! Generates the next random number.
	//! \note ONLY WORKS FOR 1D PROBLEMS
        virtual real generate(generator::Generator& generator, size_t component);
    private:
	size_t currentSample = 0;
	real deltaX;
	real a;
	real b;
    };
} // namespace generator
} // namespace alsuq
