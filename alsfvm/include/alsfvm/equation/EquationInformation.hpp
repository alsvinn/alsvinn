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
#include <string>
namespace alsfvm {
namespace equation {

///
/// Simple holder class for equations, as an equation can not be held in a
/// tuple (there is not always a default constructor available)
///
template<class T>
class EquationInformation {
public:
    typedef T EquationType;

    ///
    /// Gets the name of the equation held in the EquationInformationType.
    ///
    static std::string getName() {

        return EquationType::getName();
    }
};
} // namespace alsfvm
} // namespace equation
