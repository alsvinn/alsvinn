#pragma once
#include "alsfvm/integrator/Integrator.hpp"
#include "alsfvm/numflux/NumericalFlux.hpp"
#include "alsfvm/integrator/System.hpp"

namespace alsfvm {
namespace integrator {

class IntegratorFactory {
public:
    IntegratorFactory(const std::string& integratorName);
    alsfvm::shared_ptr<Integrator> createIntegrator(alsfvm::shared_ptr<System>&
        system);


private:
    std::string integratorName;

};
} // namespace alsfvm
} // namespace integrator
