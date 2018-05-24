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
#include "alsuq/generator/Generator.hpp"
namespace alsuq {
namespace generator {

//! \note This class is a singleton
class Well512A : public Generator {
public:
    //! Gets the one instance of the Well512 generator
    static std::shared_ptr<Generator> getInstance();

    //! Generates the next random number
    real generate(size_t component);
private:
    // Singleton
    Well512A();

    static constexpr size_t R = 16;
    unsigned int state_i = 0;
    unsigned int STATE[R];
    unsigned int z0, z1, z2;
};
} // namespace generator
} // namespace alsuq
