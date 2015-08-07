#pragma once
#include "alsfvm/integrator/Integrator.hpp"
#include "alsfvm/numflux/NumericalFlux.hpp"
namespace alsfvm { namespace integrator { 

    class IntegratorFactory {
    public:
        IntegratorFactory(const std::string& integratorName);
        std::shared_ptr<Integrator> createIntegrator(std::shared_ptr<numflux::NumericalFlux>& numericalFlux);


    private:
        std::string integratorName;

    };
} // namespace alsfvm
} // namespace integrator
