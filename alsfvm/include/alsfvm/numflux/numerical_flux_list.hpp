#pragma once

#include <boost/fusion/container.hpp>
#include <boost/fusion/sequence/intrinsic/at_key.hpp>
#include "alsfvm/equation/equation_list.hpp"
#include "alsfvm/numflux/euler/HLL.hpp"
#include "alsfvm/numflux/euler/HLL3.hpp"
#include "alsfvm/numflux/Central.hpp"


///
/// This file provides the list of all available fluxes for each equation
///
namespace alsfvm {
namespace numflux {
    typedef boost::fusion::map<
        // EULER
        boost::fusion::pair<equation::euler::Euler,
        boost::fusion::vector<
            euler::HLL,
            euler::HLL3,
            Central<equation::euler::Euler>
        >>,

        // BURGERS
        boost::fusion::pair<equation::burgers::Burgers,
        boost::fusion::vector<
            Central<equation::burgers::Burgers>
        >>
        > NumericalFluxList;


    template<class Equation, class Function>
    void for_each_flux(Function f) {

        NumericalFluxList map;


        boost::fusion::for_each(boost::fusion::at_key<Equation>(map), f);
    }



}
}

#define ALSFVM_FLUX_INSTANTIATE(X) \
    template class X<::alsfvm::numflux::euler::HLL, ::alsfvm::equation::euler::Euler,  1>; \
    template class X<::alsfvm::numflux::euler::HLL, ::alsfvm::equation::euler::Euler,  2>; \
    template class X<::alsfvm::numflux::euler::HLL, ::alsfvm::equation::euler::Euler,  3>; \
    template class X<::alsfvm::numflux::euler::HLL3, ::alsfvm::equation::euler::Euler, 1>; \
    template class X<::alsfvm::numflux::euler::HLL3, ::alsfvm::equation::euler::Euler, 2>; \
    template class X<::alsfvm::numflux::euler::HLL3, ::alsfvm::equation::euler::Euler, 3>; \
    template class X<::alsfvm::numflux::Central<equation::euler::Euler>, ::alsfvm::equation::euler::Euler, 1>; \
    template class X<::alsfvm::numflux::Central<equation::euler::Euler>, ::alsfvm::equation::euler::Euler, 2>; \
    template class X<::alsfvm::numflux::Central<equation::euler::Euler>, ::alsfvm::equation::euler::Euler, 3>; \
    template class X<::alsfvm::numflux::Central<equation::burgers::Burgers>, ::alsfvm::equation::burgers::Burgers, 1>; \
    template class X<::alsfvm::numflux::Central<equation::burgers::Burgers>, ::alsfvm::equation::burgers::Burgers, 2>; \
    template class X<::alsfvm::numflux::Central<equation::burgers::Burgers>, ::alsfvm::equation::burgers::Burgers, 3>;
