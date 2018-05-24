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
#include "alsfvm/functional/Functional.hpp"

namespace alsfvm {
namespace functional {

//! Factory class for creating a functional
class FunctionalFactory {
public:
    typedef Functional::Parameters Parameters;
    typedef std::function<FunctionalPointer(const Parameters&)> FunctionalCreator;

    static void registerFunctional(const std::string& platform,
        const std::string& name,
        FunctionalFactory::FunctionalCreator maker);

    FunctionalPointer makeFunctional(const std::string& platform,
        const std::string& name,
        const FunctionalFactory::Parameters& parameters);
};
} // namespace functional
} // namespace alsfvm
