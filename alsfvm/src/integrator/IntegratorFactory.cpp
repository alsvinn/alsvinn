#include "alsfvm/integrator/IntegratorFactory.hpp"
#include "alsfvm/integrator/ForwardEuler.hpp"
#include "alsfvm/integrator/RungeKutta2.hpp"
#include "alsfvm/integrator/RungeKutta3.hpp"
#include "alsfvm/error/Exception.hpp"

namespace alsfvm { namespace integrator {

IntegratorFactory::IntegratorFactory(const std::string &integratorName)
    : integratorName(integratorName)
{

}

alsfvm::shared_ptr<Integrator> IntegratorFactory::createIntegrator(alsfvm::shared_ptr<numflux::NumericalFlux> &numericalFlux)
{
    if (integratorName == "forwardeuler") {
        return alsfvm::shared_ptr<Integrator>(new ForwardEuler(numericalFlux));
    } else if (integratorName == "rungekutta2") {
        return alsfvm::shared_ptr<Integrator>(new RungeKutta2(numericalFlux));
    } else if (integratorName == "rungekutta3") {
        return boost::shared_ptr<Integrator>(new RungeKutta3(numericalFlux));
    } else {
        THROW("Unknown integrator " << integratorName);
    }
}

}
}
