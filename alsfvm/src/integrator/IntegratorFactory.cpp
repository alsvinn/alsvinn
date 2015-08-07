#include "alsfvm/integrator/IntegratorFactory.hpp"
#include "alsfvm/integrator/ForwardEuler.hpp"
#include "alsfvm/integrator/RungeKutta2.hpp"
#include "alsfvm/error/Exception.hpp"

namespace alsfvm { namespace integrator {

IntegratorFactory::IntegratorFactory(const std::string &integratorName)
    : integratorName(integratorName)
{

}

std::shared_ptr<Integrator> IntegratorFactory::createIntegrator(std::shared_ptr<numflux::NumericalFlux> &numericalFlux)
{
    if (integratorName == "forwardeuler") {
        return std::shared_ptr<Integrator>(new ForwardEuler(numericalFlux));
    } else if (integratorName == "rungekutta2") {
        return std::shared_ptr<Integrator>(new RungeKutta2(numericalFlux));
    } else {
        THROW("Unknown integrator " << integratorName);
    }
}

}
}
