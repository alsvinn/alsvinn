#include "alsuq/distribution/Uniform1D.hpp"
#include "alsutils/error/Exception.hpp"
namespace alsuq { namespace distribution {
	Uniform1D::Uniform1D(size_t numberOfSamples, double a, double b)
	    : a(a), b(b)
	{
	    deltaX = (b-a)/numberOfSamples;
	}

	real Uniform1D::generate(generator::Generator& generator, size_t component) {
	    if (component > 0) {
		THROW("With Uniform1D we only support 1 dimension");
	    }

	    real midpoint = a + deltaX * (currentSample + 0.5);
	    currentSample++;
	    return midpoint;
	}
		
}
}
