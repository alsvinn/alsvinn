/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "alsfvm/integrator/IntegratorFactory.hpp"
#include "alsfvm/integrator/ForwardEuler.hpp"
#include "alsfvm/integrator/RungeKutta2.hpp"
#include "alsfvm/integrator/RungeKutta3.hpp"
#include "alsfvm/integrator/RungeKutta4.hpp"
#include "alsutils/error/Exception.hpp"

namespace alsfvm {
namespace integrator {

IntegratorFactory::IntegratorFactory(const std::string& integratorName)
    : integratorName(integratorName) {

}

alsfvm::shared_ptr<Integrator> IntegratorFactory::createIntegrator(
    alsfvm::shared_ptr<System>& system) {
    if (integratorName == "forwardeuler") {
        return alsfvm::shared_ptr<Integrator>(new ForwardEuler(system));
    } else if (integratorName == "rungekutta2") {
        return alsfvm::shared_ptr<Integrator>(new RungeKutta2(system));
    } else if (integratorName == "rungekutta3") {
        return alsfvm::shared_ptr<Integrator>(new RungeKutta3(system));
    } else if (integratorName == "rungekutta4") {
        return alsfvm::shared_ptr<Integrator>(new RungeKutta4(system));
    } else {
        THROW("Unknown integrator " << integratorName);
    }
}

}
}
