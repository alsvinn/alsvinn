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

#include "alsfvm/numflux/Rusanov.hpp"
#include "alsfvm/equation/equation_list.hpp"
namespace alsfvm {
namespace numflux {
template<class Equation>
const std::string Rusanov<Equation>::name = "rusanov";

template class Rusanov<equation::burgers::Burgers>;
template class Rusanov<equation::buckleyleverett::BuckleyLeverett>;
template class Rusanov<equation::cubic::Cubic>;
template class Rusanov<equation::linear::Linear>;
}
}
