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
