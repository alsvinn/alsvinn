#include "alsfvm/numflux/TecnoCombined4.hpp"
#include "alsfvm/numflux/ScalarEntropyConservativeFlux.hpp"
#include "alsfvm/equation/equation_list.hpp"
namespace alsfvm { namespace numflux { 
    template<class Equation, class BaseFlux>
    const std::string TecnoCombined4<Equation, BaseFlux >::name = "tecno4";

    template class TecnoCombined4< ::alsfvm::equation::euler::Euler, ScalarEntropyConservativeFlux<equation::euler::Euler>>;
    template class TecnoCombined4< ::alsfvm::equation::burgers::Burgers, ScalarEntropyConservativeFlux<equation::burgers::Burgers> >;
}
}
