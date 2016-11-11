#include "alsfvm/numflux/TecnoCombined6.hpp"
#include "alsfvm/numflux/ScalarEntropyConservativeFlux.hpp"
#include "alsfvm/equation/equation_list.hpp"
namespace alsfvm { namespace numflux {

    template<>
    const std::string TecnoCombined6<::alsfvm::equation::burgers::Burgers, ScalarEntropyConservativeFlux<equation::burgers::Burgers > >::name = "tecno6";

    template<>
    const std::string TecnoCombined6<::alsfvm::equation::euler::Euler, ScalarEntropyConservativeFlux<equation::euler::Euler > >::name = "tecno6";

    template<>
    const std::string TecnoCombined6<::alsfvm::equation::burgers::Burgers, burgers::Godunov>::name = "godunov6";


    template class TecnoCombined6<::alsfvm::equation::burgers::Burgers, ScalarEntropyConservativeFlux<equation::burgers::Burgers > >;
    template class TecnoCombined6<::alsfvm::equation::euler::Euler, ScalarEntropyConservativeFlux<equation::euler::Euler > >;
    template class TecnoCombined6<::alsfvm::equation::burgers::Burgers, burgers::Godunov>;
}
}
