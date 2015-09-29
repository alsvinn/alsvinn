#pragma once
#include "alsfvm/integrator/Integrator.hpp"
#include "alsfvm/numflux/NumericalFlux.hpp"
namespace alsfvm { namespace integrator { 

    class IntegratorFactory {
    public:
        IntegratorFactory(const std::string& integratorName);
        boost::shared_ptr<Integrator> createIntegrator(boost::shared_ptr<numflux::NumericalFlux>& numericalFlux);


    private:
        std::string integratorName;

    };
} // namespace alsfvm
} // namespace integrator
