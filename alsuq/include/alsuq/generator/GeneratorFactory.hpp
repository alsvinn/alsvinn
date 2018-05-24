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

class GeneratorFactory {
public:

    //!
    //! \brief makeGenerator creates a new generator
    //! \param name the name of the generator
    //! \param dimensions the number of dimensions to use
    //! \param numberVariables number of random variables to draw (relevant for QMC)
    //! \return the new generator
    //!
    std::shared_ptr<Generator> makeGenerator(const std::string& name,
        const size_t dimensions,
        const size_t numberVariables
    );


};
} // namespace generator
} // namespace alsuq
