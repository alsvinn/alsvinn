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

#include "alsfvm/equation/equation_list.hpp"
#include "alsfvm/numflux/euler/HLL.hpp"
#include "alsfvm/numflux/euler/HLL3.hpp"
#include "alsfvm/numflux/burgers/Godunov.hpp"
#include "alsfvm/numflux/buckleyleverett/Godunov.hpp"
#include "alsfvm/numflux/Central.hpp"
#include "alsfvm/numflux/Rusanov.hpp"
#include "alsfvm/numflux/ScalarEntropyConservativeFlux.hpp"
#include "alsfvm/numflux/TecnoCombined4.hpp"
#include "alsfvm/numflux/TecnoCombined6.hpp"
#include "alsfvm/numflux/euler/Tecno1.hpp"
#include "alsfvm/equation/linear/Linear.hpp"
#include "alsfvm/numflux/linear/Upwind.hpp"


///
/// This file provides the list of all available fluxes for each equation
///
namespace alsfvm {
namespace numflux {
typedef boost::fusion::map <
// EULER1
boost::fusion::pair<equation::euler::Euler<1>,
      boost::fusion::vector<
      euler::HLL<1>,
      euler::HLL3<1>,
      Central<equation::euler::Euler<1>>,
      euler::Tecno1<1>,
      TecnoCombined4<equation::euler::Euler<1>, euler::Tecno1<1> >,
      TecnoCombined6<equation::euler::Euler<1>, euler::Tecno1<1> >
      > >,

      // EULER2
      boost::fusion::pair<equation::euler::Euler<2>,
      boost::fusion::vector<
      euler::HLL<2>,
      euler::HLL3<2>,
      Central<equation::euler::Euler<2>>,
      euler::Tecno1<2>,
      TecnoCombined4<equation::euler::Euler<2>, euler::Tecno1<2> >,
      TecnoCombined6<equation::euler::Euler<2>, euler::Tecno1<2> >
      > >,

      // EULER3
      boost::fusion::pair<equation::euler::Euler<3>,
      boost::fusion::vector<
      euler::HLL<3>,
      euler::HLL3<3>,
      Central<equation::euler::Euler<3>>,
      euler::Tecno1<3>,
      TecnoCombined4<equation::euler::Euler<3>, euler::Tecno1<3> >,
      TecnoCombined6<equation::euler::Euler<3>, euler::Tecno1<3> >
      > >,

      // BURGERS
      boost::fusion::pair < equation::burgers::Burgers,
      boost::fusion::vector <
      Central<equation::burgers::Burgers>,
      Rusanov<equation::burgers::Burgers>,
      burgers::Godunov,
      ScalarEntropyConservativeFlux<equation::burgers::Burgers>,
      TecnoCombined4<equation::burgers::Burgers, ScalarEntropyConservativeFlux<equation::burgers::Burgers> >,
      TecnoCombined4<::alsfvm::equation::burgers::Burgers, burgers::Godunov>,
      TecnoCombined6<equation::burgers::Burgers, ScalarEntropyConservativeFlux<equation::burgers::Burgers> >,
      TecnoCombined6<::alsfvm::equation::burgers::Burgers, burgers::Godunov>
      >>,

      // BUCKLEY-LEVERETT
      boost::fusion::pair < equation::buckleyleverett::BuckleyLeverett,
      boost::fusion::vector <
      Central<equation::buckleyleverett::BuckleyLeverett>,
      buckleyleverett::Godunov,
      Rusanov<equation::buckleyleverett::BuckleyLeverett>
      > >,

      // CUBIC
      boost::fusion::pair < equation::cubic::Cubic,
      boost::fusion::vector <
      Central<equation::cubic::Cubic>,
      Rusanov<equation::cubic::Cubic>
      > >,

      // Linear
      boost::fusion::pair < equation::linear::Linear,
      boost::fusion::vector <
      Central<equation::linear::Linear>,
      Rusanov<equation::linear::Linear>,
      linear::Upwind
      > >
      > NumericalFluxList;


template<class Equation, class Function>
void for_each_flux(Function f) {

    NumericalFluxList map;


    boost::fusion::for_each(boost::fusion::at_key<Equation>(map), f);
}



}
}

#define ALSFVM_FLUX_INSTANTIATE(X) \
    template class X< ::alsfvm::numflux::euler::HLL<1>, ::alsfvm::equation::euler::Euler<1>,  1>; \
    template class X< ::alsfvm::numflux::euler::HLL<1>, ::alsfvm::equation::euler::Euler<1>,  2>; \
    template class X< ::alsfvm::numflux::euler::HLL<1>, ::alsfvm::equation::euler::Euler<1>,  3>; \
    template class X< ::alsfvm::numflux::euler::HLL<2>, ::alsfvm::equation::euler::Euler<2>,  1>; \
    template class X< ::alsfvm::numflux::euler::HLL<2>, ::alsfvm::equation::euler::Euler<2>,  2>; \
    template class X< ::alsfvm::numflux::euler::HLL<2>, ::alsfvm::equation::euler::Euler<2>,  3>; \
    template class X< ::alsfvm::numflux::euler::HLL<3>, ::alsfvm::equation::euler::Euler<3>,  1>; \
    template class X< ::alsfvm::numflux::euler::HLL<3>, ::alsfvm::equation::euler::Euler<3>,  2>; \
    template class X< ::alsfvm::numflux::euler::HLL<3>, ::alsfvm::equation::euler::Euler<3>,  3>; \
    template class X< ::alsfvm::numflux::euler::HLL3<1>, ::alsfvm::equation::euler::Euler<1>, 1>; \
    template class X< ::alsfvm::numflux::euler::HLL3<1>, ::alsfvm::equation::euler::Euler<1>, 2>; \
    template class X< ::alsfvm::numflux::euler::HLL3<1>, ::alsfvm::equation::euler::Euler<1>, 3>; \
    template class X< ::alsfvm::numflux::euler::HLL3<2>, ::alsfvm::equation::euler::Euler<2>, 1>; \
    template class X< ::alsfvm::numflux::euler::HLL3<2>, ::alsfvm::equation::euler::Euler<2>, 2>; \
    template class X< ::alsfvm::numflux::euler::HLL3<2>, ::alsfvm::equation::euler::Euler<2>, 3>; \
    template class X< ::alsfvm::numflux::euler::HLL3<3>, ::alsfvm::equation::euler::Euler<3>, 1>; \
    template class X< ::alsfvm::numflux::euler::HLL3<3>, ::alsfvm::equation::euler::Euler<3>, 2>; \
    template class X< ::alsfvm::numflux::euler::HLL3<3>, ::alsfvm::equation::euler::Euler<3>, 3>; \
    template class X< ::alsfvm::numflux::Central<equation::euler::Euler<1>>, ::alsfvm::equation::euler::Euler<1>, 1>; \
    template class X< ::alsfvm::numflux::Central<equation::euler::Euler<1>>, ::alsfvm::equation::euler::Euler<1>, 2>; \
    template class X< ::alsfvm::numflux::Central<equation::euler::Euler<1>>, ::alsfvm::equation::euler::Euler<1>, 3>; \
    template class X< ::alsfvm::numflux::Central<equation::euler::Euler<2>>, ::alsfvm::equation::euler::Euler<2>, 1>; \
    template class X< ::alsfvm::numflux::Central<equation::euler::Euler<2>>, ::alsfvm::equation::euler::Euler<2>, 2>; \
    template class X< ::alsfvm::numflux::Central<equation::euler::Euler<2>>, ::alsfvm::equation::euler::Euler<2>, 3>; \
    template class X< ::alsfvm::numflux::Central<equation::euler::Euler<3>>, ::alsfvm::equation::euler::Euler<3>, 1>; \
    template class X< ::alsfvm::numflux::Central<equation::euler::Euler<3>>, ::alsfvm::equation::euler::Euler<3>, 2>; \
    template class X< ::alsfvm::numflux::Central<equation::euler::Euler<3>>, ::alsfvm::equation::euler::Euler<3>, 3>; \
    template class X< ::alsfvm::numflux::Central<equation::burgers::Burgers>, ::alsfvm::equation::burgers::Burgers, 1>; \
    template class X< ::alsfvm::numflux::Central<equation::burgers::Burgers>, ::alsfvm::equation::burgers::Burgers, 2>; \
    template class X< ::alsfvm::numflux::Central<equation::burgers::Burgers>, ::alsfvm::equation::burgers::Burgers, 3>; \
    template class X< ::alsfvm::numflux::Rusanov<equation::burgers::Burgers>, ::alsfvm::equation::burgers::Burgers, 1>; \
    template class X< ::alsfvm::numflux::Rusanov<equation::burgers::Burgers>, ::alsfvm::equation::burgers::Burgers, 2>; \
    template class X< ::alsfvm::numflux::Rusanov<equation::burgers::Burgers>, ::alsfvm::equation::burgers::Burgers, 3>; \
    template class X< ::alsfvm::numflux::burgers::Godunov, ::alsfvm::equation::burgers::Burgers, 1>; \
    template class X< ::alsfvm::numflux::burgers::Godunov, ::alsfvm::equation::burgers::Burgers, 2>; \
    template class X< ::alsfvm::numflux::burgers::Godunov, ::alsfvm::equation::burgers::Burgers, 3>; \
    template class X< ::alsfvm::numflux::ScalarEntropyConservativeFlux<::alsfvm::equation::burgers::Burgers>, ::alsfvm::equation::burgers::Burgers, 1>; \
    template class X< ::alsfvm::numflux::ScalarEntropyConservativeFlux<::alsfvm::equation::burgers::Burgers>, ::alsfvm::equation::burgers::Burgers, 2>; \
    template class X< ::alsfvm::numflux::ScalarEntropyConservativeFlux<::alsfvm::equation::burgers::Burgers>, ::alsfvm::equation::burgers::Burgers, 3>; \
    template class X< ::alsfvm::numflux::TecnoCombined4<::alsfvm::equation::burgers::Burgers, ::alsfvm::numflux::ScalarEntropyConservativeFlux<::alsfvm::equation::burgers::Burgers> >, ::alsfvm::equation::burgers::Burgers, 1>; \
    template class X< ::alsfvm::numflux::TecnoCombined4<::alsfvm::equation::burgers::Burgers, ::alsfvm::numflux::ScalarEntropyConservativeFlux<::alsfvm::equation::burgers::Burgers> >, ::alsfvm::equation::burgers::Burgers, 2>; \
    template class X< ::alsfvm::numflux::TecnoCombined4<::alsfvm::equation::burgers::Burgers, ::alsfvm::numflux::ScalarEntropyConservativeFlux<::alsfvm::equation::burgers::Burgers> >, ::alsfvm::equation::burgers::Burgers, 3>; \
    template class X< ::alsfvm::numflux::TecnoCombined4<::alsfvm::equation::burgers::Burgers, ::alsfvm::numflux::burgers::Godunov >, ::alsfvm::equation::burgers::Burgers, 1>; \
    template class X< ::alsfvm::numflux::TecnoCombined4<::alsfvm::equation::burgers::Burgers, ::alsfvm::numflux::burgers::Godunov >, ::alsfvm::equation::burgers::Burgers, 2>; \
    template class X< ::alsfvm::numflux::TecnoCombined4<::alsfvm::equation::burgers::Burgers, ::alsfvm::numflux::burgers::Godunov >, ::alsfvm::equation::burgers::Burgers, 3>; \
    template class X< ::alsfvm::numflux::TecnoCombined6<::alsfvm::equation::burgers::Burgers, ::alsfvm::numflux::ScalarEntropyConservativeFlux<::alsfvm::equation::burgers::Burgers> >, ::alsfvm::equation::burgers::Burgers, 1>; \
    template class X< ::alsfvm::numflux::TecnoCombined6<::alsfvm::equation::burgers::Burgers, ::alsfvm::numflux::ScalarEntropyConservativeFlux<::alsfvm::equation::burgers::Burgers> >, ::alsfvm::equation::burgers::Burgers, 2>; \
    template class X< ::alsfvm::numflux::TecnoCombined6<::alsfvm::equation::burgers::Burgers, ::alsfvm::numflux::ScalarEntropyConservativeFlux<::alsfvm::equation::burgers::Burgers> >, ::alsfvm::equation::burgers::Burgers, 3>; \
    template class X< ::alsfvm::numflux::TecnoCombined6<::alsfvm::equation::burgers::Burgers, ::alsfvm::numflux::burgers::Godunov >, ::alsfvm::equation::burgers::Burgers, 1>; \
    template class X< ::alsfvm::numflux::TecnoCombined6<::alsfvm::equation::burgers::Burgers, ::alsfvm::numflux::burgers::Godunov >, ::alsfvm::equation::burgers::Burgers, 2>; \
    template class X< ::alsfvm::numflux::TecnoCombined6<::alsfvm::equation::burgers::Burgers, ::alsfvm::numflux::burgers::Godunov >, ::alsfvm::equation::burgers::Burgers, 3>; \
    template class X< ::alsfvm::numflux::euler::Tecno1<1>, ::alsfvm::equation::euler::Euler<1>, 1>; \
    template class X< ::alsfvm::numflux::euler::Tecno1<1>, ::alsfvm::equation::euler::Euler<1>, 2>; \
    template class X< ::alsfvm::numflux::euler::Tecno1<1>, ::alsfvm::equation::euler::Euler<1>, 3>; \
    template class X< ::alsfvm::numflux::euler::Tecno1<2>, ::alsfvm::equation::euler::Euler<2>, 1>; \
    template class X< ::alsfvm::numflux::euler::Tecno1<2>, ::alsfvm::equation::euler::Euler<2>, 2>; \
    template class X< ::alsfvm::numflux::euler::Tecno1<2>, ::alsfvm::equation::euler::Euler<2>, 3>; \
    template class X< ::alsfvm::numflux::euler::Tecno1<3>, ::alsfvm::equation::euler::Euler<3>, 1>; \
    template class X< ::alsfvm::numflux::euler::Tecno1<3>, ::alsfvm::equation::euler::Euler<3>, 2>; \
    template class X< ::alsfvm::numflux::euler::Tecno1<3>, ::alsfvm::equation::euler::Euler<3>, 3>; \
    template class X< ::alsfvm::numflux::TecnoCombined4<::alsfvm::equation::euler::Euler<1>, ::alsfvm::numflux::euler::Tecno1<1> >, ::alsfvm::equation::euler::Euler<1>, 1>; \
    template class X< ::alsfvm::numflux::TecnoCombined4<::alsfvm::equation::euler::Euler<1>, ::alsfvm::numflux::euler::Tecno1<1> >, ::alsfvm::equation::euler::Euler<1>, 2>; \
    template class X< ::alsfvm::numflux::TecnoCombined4<::alsfvm::equation::euler::Euler<1>, ::alsfvm::numflux::euler::Tecno1<1> >, ::alsfvm::equation::euler::Euler<1>, 3>; \
    template class X< ::alsfvm::numflux::TecnoCombined4<::alsfvm::equation::euler::Euler<2>, ::alsfvm::numflux::euler::Tecno1<2> >, ::alsfvm::equation::euler::Euler<2>, 1>; \
    template class X< ::alsfvm::numflux::TecnoCombined4<::alsfvm::equation::euler::Euler<2>, ::alsfvm::numflux::euler::Tecno1<2> >, ::alsfvm::equation::euler::Euler<2>, 2>; \
    template class X< ::alsfvm::numflux::TecnoCombined4<::alsfvm::equation::euler::Euler<2>, ::alsfvm::numflux::euler::Tecno1<2> >, ::alsfvm::equation::euler::Euler<2>, 3>; \
    template class X< ::alsfvm::numflux::TecnoCombined4<::alsfvm::equation::euler::Euler<3>, ::alsfvm::numflux::euler::Tecno1<3> >, ::alsfvm::equation::euler::Euler<3>, 1>; \
    template class X< ::alsfvm::numflux::TecnoCombined4<::alsfvm::equation::euler::Euler<3>, ::alsfvm::numflux::euler::Tecno1<3> >, ::alsfvm::equation::euler::Euler<3>, 2>; \
    template class X< ::alsfvm::numflux::TecnoCombined4<::alsfvm::equation::euler::Euler<3>, ::alsfvm::numflux::euler::Tecno1<3> >, ::alsfvm::equation::euler::Euler<3>, 3>; \
    template class X< ::alsfvm::numflux::TecnoCombined6<::alsfvm::equation::euler::Euler<1>, ::alsfvm::numflux::euler::Tecno1<1> >, ::alsfvm::equation::euler::Euler<1>, 1>; \
    template class X< ::alsfvm::numflux::TecnoCombined6<::alsfvm::equation::euler::Euler<1>, ::alsfvm::numflux::euler::Tecno1<1> >, ::alsfvm::equation::euler::Euler<1>, 2>; \
    template class X< ::alsfvm::numflux::TecnoCombined6<::alsfvm::equation::euler::Euler<1>, ::alsfvm::numflux::euler::Tecno1<1> >, ::alsfvm::equation::euler::Euler<1>, 3>; \
    template class X< ::alsfvm::numflux::TecnoCombined6<::alsfvm::equation::euler::Euler<2>, ::alsfvm::numflux::euler::Tecno1<2> >, ::alsfvm::equation::euler::Euler<2>, 1>; \
    template class X< ::alsfvm::numflux::TecnoCombined6<::alsfvm::equation::euler::Euler<2>, ::alsfvm::numflux::euler::Tecno1<2> >, ::alsfvm::equation::euler::Euler<2>, 2>; \
    template class X< ::alsfvm::numflux::TecnoCombined6<::alsfvm::equation::euler::Euler<2>, ::alsfvm::numflux::euler::Tecno1<2> >, ::alsfvm::equation::euler::Euler<2>, 3>; \
    template class X< ::alsfvm::numflux::TecnoCombined6<::alsfvm::equation::euler::Euler<3>, ::alsfvm::numflux::euler::Tecno1<3> >, ::alsfvm::equation::euler::Euler<3>, 1>; \
    template class X< ::alsfvm::numflux::TecnoCombined6<::alsfvm::equation::euler::Euler<3>, ::alsfvm::numflux::euler::Tecno1<3> >, ::alsfvm::equation::euler::Euler<3>, 2>; \
    template class X< ::alsfvm::numflux::TecnoCombined6<::alsfvm::equation::euler::Euler<3>, ::alsfvm::numflux::euler::Tecno1<3> >, ::alsfvm::equation::euler::Euler<3>, 3>; \
    template class X< ::alsfvm::numflux::Central<equation::buckleyleverett::BuckleyLeverett>, ::alsfvm::equation::buckleyleverett::BuckleyLeverett, 1>; \
    template class X< ::alsfvm::numflux::Central<equation::buckleyleverett::BuckleyLeverett>, ::alsfvm::equation::buckleyleverett::BuckleyLeverett, 2>; \
    template class X< ::alsfvm::numflux::Central<equation::buckleyleverett::BuckleyLeverett>, ::alsfvm::equation::buckleyleverett::BuckleyLeverett, 3>; \
    template class X< ::alsfvm::numflux::Rusanov<equation::buckleyleverett::BuckleyLeverett>, ::alsfvm::equation::buckleyleverett::BuckleyLeverett, 1>; \
    template class X< ::alsfvm::numflux::Rusanov<equation::buckleyleverett::BuckleyLeverett>, ::alsfvm::equation::buckleyleverett::BuckleyLeverett, 2>; \
    template class X< ::alsfvm::numflux::Rusanov<equation::buckleyleverett::BuckleyLeverett>, ::alsfvm::equation::buckleyleverett::BuckleyLeverett, 3>; \
    template class X< ::alsfvm::numflux::buckleyleverett::Godunov, ::alsfvm::equation::buckleyleverett::BuckleyLeverett, 1>; \
    template class X< ::alsfvm::numflux::buckleyleverett::Godunov, ::alsfvm::equation::buckleyleverett::BuckleyLeverett, 2>; \
    template class X< ::alsfvm::numflux::buckleyleverett::Godunov, ::alsfvm::equation::buckleyleverett::BuckleyLeverett, 3>; \
    template class X< ::alsfvm::numflux::Central<equation::cubic::Cubic>, ::alsfvm::equation::cubic::Cubic, 1>; \
    template class X< ::alsfvm::numflux::Central<equation::cubic::Cubic>, ::alsfvm::equation::cubic::Cubic, 2>; \
    template class X< ::alsfvm::numflux::Central<equation::cubic::Cubic>, ::alsfvm::equation::cubic::Cubic, 3>; \
    template class X< ::alsfvm::numflux::Rusanov<equation::cubic::Cubic>, ::alsfvm::equation::cubic::Cubic, 1>; \
    template class X< ::alsfvm::numflux::Rusanov<equation::cubic::Cubic>, ::alsfvm::equation::cubic::Cubic, 2>; \
    template class X< ::alsfvm::numflux::Rusanov<equation::cubic::Cubic>, ::alsfvm::equation::cubic::Cubic, 3>; \
    template class X< ::alsfvm::numflux::Central<equation::linear::Linear>, ::alsfvm::equation::linear::Linear, 1>; \
    template class X< ::alsfvm::numflux::Central<equation::linear::Linear>, ::alsfvm::equation::linear::Linear, 2>; \
    template class X< ::alsfvm::numflux::Central<equation::linear::Linear>, ::alsfvm::equation::linear::Linear, 3>; \
    template class X< ::alsfvm::numflux::Rusanov<equation::linear::Linear>, ::alsfvm::equation::linear::Linear, 1>; \
    template class X< ::alsfvm::numflux::Rusanov<equation::linear::Linear>, ::alsfvm::equation::linear::Linear, 2>; \
    template class X< ::alsfvm::numflux::Rusanov<equation::linear::Linear>, ::alsfvm::equation::linear::Linear, 3>; \
    template class X< ::alsfvm::numflux::linear::Upwind, ::alsfvm::equation::linear::Linear, 1>; \
    template class X< ::alsfvm::numflux::linear::Upwind, ::alsfvm::equation::linear::Linear, 2>; \
    template class X< ::alsfvm::numflux::linear::Upwind, ::alsfvm::equation::linear::Linear, 3>;




