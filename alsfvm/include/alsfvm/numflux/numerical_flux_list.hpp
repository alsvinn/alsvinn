#pragma once

#include <boost/fusion/container.hpp>
#include <boost/fusion/sequence/intrinsic/at_key.hpp>
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
    template class X< ::alsfvm::numflux::Rusanov<equation::cubic::Cubic>, ::alsfvm::equation::cubic::Cubic, 3>;




