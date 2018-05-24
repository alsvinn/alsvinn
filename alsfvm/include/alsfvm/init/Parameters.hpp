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
#include <vector>
#include <string>
#include <map>
#include "alsfvm/types.hpp"
namespace alsfvm {
namespace init {

//! Parameters for the initial data.
//! These are typically used to give random
//! inputs.
class Parameters {
public:

    //! Add a a parameter to the parameters.
    void addParameter(const std::string& name, const std::vector<real>& value);


    std::vector<std::string> getParameterNames() const;

    //! Each parameter is represented by an array
    //! A scalar is then represented by a length one array.
    const std::vector<real>& getParameter(const std::string& name) const;

private:
    std::map<std::string,  std::vector<real> > parameters;
};
} // namespace init
} // namespace alsfvm
