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
#include "alsfvm/types.hpp"
namespace alsfvm {
namespace reconstruction {

template<int k>
class ENOCoeffiecients {
public:
    ///
    /// \brief coefficients are the ENO coefficients.
    ///
    /// coefficients[r][i] indexes the coefficient with shift r and term i
    ///
    /// See http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19980007543.pdf
    /// for description.
    ///
    static real coefficients[][k];
};
}
}
