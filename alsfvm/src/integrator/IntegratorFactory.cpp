#include "alsfvm/integrator/IntegratorFactory.hpp"
#include "alsfvm/integrator/ForwardEuler.hpp"
#include "alsfvm/integrator/RungeKutta2.hpp"
#include "alsfvm/integrator/RungeKutta3.hpp"
#include "alsfvm/integrator/RungeKutta4.hpp"
#include "alsfvm/error/Exception.hpp"

namespace alsfvm { namespace integrator {

IntegratorFactory::IntegratorFactory(const std::string &integratorName)
    : integratorName(integratorName)
{

}

alsfvm::shared_ptr<Integrator> IntegratorFactory::createIntegrator(alsfvm::shared_ptr<System> &system)
{
    if (integratorName == "forwardeuler") {
        return alsfvm::shared_ptr<Integrator>(new ForwardEuler(system));
    } else if (integratorName == "rungekutta2") {
        return alsfvm::shared_ptr<Integrator>(new RungeKutta2(system));
    } else if (integratorName == "rungekutta3") {
        return alsfvm::shared_ptr<Integrator>(new RungeKutta3(system));
    } 
    else if (integratorName == "rungekutta4") {
        return alsfvm::shared_ptr<Integrator>(new RungeKutta4(system));
    }
    else {
        THROW("Unknown integrator " << integratorName);
    }
}

}
}
