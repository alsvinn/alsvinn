#include "alsfvm/numflux/TecnoCombined4.hpp"
#include "alsfvm/numflux/ScalarEntropyConservativeFlux.hpp"
#include "alsfvm/equation/equation_list.hpp"
namespace alsfvm { namespace numflux { 

    template<>
    const std::string TecnoCombined4<::alsfvm::equation::burgers::Burgers, ScalarEntropyConservativeFlux<equation::burgers::Burgers > >::name = "tecno4";

    template<>
    const std::string TecnoCombined4<::alsfvm::equation::euler::Euler, ScalarEntropyConservativeFlux<equation::euler::Euler > >::name = "tecno4";

    template<>
    const std::string TecnoCombined4<::alsfvm::equation::burgers::Burgers, burgers::Godunov>::name = "godunov4";


    template class TecnoCombined4<::alsfvm::equation::burgers::Burgers, ScalarEntropyConservativeFlux<equation::burgers::Burgers > >;
    template class TecnoCombined4<::alsfvm::equation::euler::Euler, ScalarEntropyConservativeFlux<equation::euler::Euler > >;
    template class TecnoCombined4<::alsfvm::equation::burgers::Burgers, burgers::Godunov>;
}
}
