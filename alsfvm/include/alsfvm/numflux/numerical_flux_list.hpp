#pragma once

#include <boost/fusion/container.hpp>
#include <boost/fusion/sequence/intrinsic/at_key.hpp>
#include "alsfvm/equation/equation_list.hpp"
#include "alsfvm/numflux/euler/HLL.hpp"
#include "alsfvm/numflux/euler/HLL3.hpp"
#include "alsfvm/numflux/burgers/Godunov.hpp"
#include "alsfvm/numflux/Central.hpp"
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
        // EULER
        boost::fusion::pair<equation::euler::Euler,
        boost::fusion::vector<
        euler::HLL,
        euler::HLL3,
        Central<equation::euler::Euler>,
        euler::Tecno1
        > >,

        // BURGERS
        boost::fusion::pair < equation::burgers::Burgers,
        boost::fusion::vector <
        Central<equation::burgers::Burgers>,
        burgers::Godunov,
        ScalarEntropyConservativeFlux<equation::burgers::Burgers>,
        TecnoCombined4<equation::burgers::Burgers, ScalarEntropyConservativeFlux<equation::burgers::Burgers> >,
        TecnoCombined4<::alsfvm::equation::burgers::Burgers, burgers::Godunov>,
        TecnoCombined6<equation::burgers::Burgers, ScalarEntropyConservativeFlux<equation::burgers::Burgers> >,
        TecnoCombined6<::alsfvm::equation::burgers::Burgers, burgers::Godunov>

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
    template class X< ::alsfvm::numflux::euler::HLL, ::alsfvm::equation::euler::Euler,  1>; \
    template class X< ::alsfvm::numflux::euler::HLL, ::alsfvm::equation::euler::Euler,  2>; \
    template class X< ::alsfvm::numflux::euler::HLL, ::alsfvm::equation::euler::Euler,  3>; \
    template class X< ::alsfvm::numflux::euler::HLL3, ::alsfvm::equation::euler::Euler, 1>; \
    template class X< ::alsfvm::numflux::euler::HLL3, ::alsfvm::equation::euler::Euler, 2>; \
    template class X< ::alsfvm::numflux::euler::HLL3, ::alsfvm::equation::euler::Euler, 3>; \
    template class X< ::alsfvm::numflux::Central<equation::euler::Euler>, ::alsfvm::equation::euler::Euler, 1>; \
    template class X< ::alsfvm::numflux::Central<equation::euler::Euler>, ::alsfvm::equation::euler::Euler, 2>; \
    template class X< ::alsfvm::numflux::Central<equation::euler::Euler>, ::alsfvm::equation::euler::Euler, 3>; \
    template class X< ::alsfvm::numflux::Central<equation::burgers::Burgers>, ::alsfvm::equation::burgers::Burgers, 1>; \
    template class X< ::alsfvm::numflux::Central<equation::burgers::Burgers>, ::alsfvm::equation::burgers::Burgers, 2>; \
    template class X< ::alsfvm::numflux::Central<equation::burgers::Burgers>, ::alsfvm::equation::burgers::Burgers, 3>; \
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
    template class X< ::alsfvm::numflux::euler::Tecno1, ::alsfvm::equation::euler::Euler, 1>; \
    template class X< ::alsfvm::numflux::euler::Tecno1, ::alsfvm::equation::euler::Euler, 2>; \
    template class X< ::alsfvm::numflux::euler::Tecno1, ::alsfvm::equation::euler::Euler, 3>; \
    template class X< ::alsfvm::numflux::TecnoCombined4<::alsfvm::equation::euler::Euler, ::alsfvm::numflux::euler::Tecno1 >, ::alsfvm::equation::euler::Euler, 1>; \
    template class X< ::alsfvm::numflux::TecnoCombined4<::alsfvm::equation::euler::Euler, ::alsfvm::numflux::euler::Tecno1 >, ::alsfvm::equation::euler::Euler, 2>; \
    template class X< ::alsfvm::numflux::TecnoCombined4<::alsfvm::equation::euler::Euler, ::alsfvm::numflux::euler::Tecno1 >, ::alsfvm::equation::euler::Euler, 3>; \
    template class X< ::alsfvm::numflux::TecnoCombined6<::alsfvm::equation::euler::Euler, ::alsfvm::numflux::euler::Tecno1 >, ::alsfvm::equation::euler::Euler, 1>; \
    template class X< ::alsfvm::numflux::TecnoCombined6<::alsfvm::equation::euler::Euler, ::alsfvm::numflux::euler::Tecno1 >, ::alsfvm::equation::euler::Euler, 2>; \
    template class X< ::alsfvm::numflux::TecnoCombined6<::alsfvm::equation::euler::Euler, ::alsfvm::numflux::euler::Tecno1 >, ::alsfvm::equation::euler::Euler, 3>; \



