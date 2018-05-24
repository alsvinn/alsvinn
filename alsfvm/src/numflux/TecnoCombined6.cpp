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

#include "alsfvm/numflux/TecnoCombined6.hpp"
#include "alsfvm/numflux/ScalarEntropyConservativeFlux.hpp"
#include "alsfvm/equation/equation_list.hpp"
#include "alsfvm/numflux/euler/Tecno1.hpp"
namespace alsfvm {
namespace numflux {

template<>
const std::string
TecnoCombined6<::alsfvm::equation::burgers::Burgers, ScalarEntropyConservativeFlux<equation::burgers::Burgers > >::name
    = "tecno6";

template<>
const std::string
TecnoCombined6<::alsfvm::equation::euler::Euler<1>, euler::Tecno1<1> >::name =
    "tecno6";

template<>
const std::string
TecnoCombined6<::alsfvm::equation::euler::Euler<2>, euler::Tecno1<2> >::name =
    "tecno6";

template<>
const std::string
TecnoCombined6<::alsfvm::equation::euler::Euler<3>, euler::Tecno1<3> >::name =
    "tecno6";

template<>
const std::string
TecnoCombined6<::alsfvm::equation::burgers::Burgers, burgers::Godunov>::name =
    "godunov6";


template class
TecnoCombined6<::alsfvm::equation::burgers::Burgers, ScalarEntropyConservativeFlux<equation::burgers::Burgers > >;
template class
TecnoCombined6<::alsfvm::equation::euler::Euler<1>, euler::Tecno1<1>>;
template class
TecnoCombined6<::alsfvm::equation::euler::Euler<2>, euler::Tecno1<2>>;
template class
TecnoCombined6<::alsfvm::equation::euler::Euler<3>, euler::Tecno1<3>>;
template class
TecnoCombined6<::alsfvm::equation::burgers::Burgers, burgers::Godunov>;
}
}
