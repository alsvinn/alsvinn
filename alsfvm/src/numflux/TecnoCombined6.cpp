#include "alsfvm/numflux/TecnoCombined6.hpp"
#include "alsfvm/numflux/ScalarEntropyConservativeFlux.hpp"
#include "alsfvm/equation/equation_list.hpp"
#include "alsfvm/numflux/euler/Tecno1.hpp"
namespace alsfvm { namespace numflux {

    template<>
    const std::string TecnoCombined6<::alsfvm::equation::burgers::Burgers, ScalarEntropyConservativeFlux<equation::burgers::Burgers > >::name = "tecno6";

    template<>
    const std::string TecnoCombined6<::alsfvm::equation::euler::Euler, euler::Tecno1 >::name = "tecno6";

    template<>
    const std::string TecnoCombined6<::alsfvm::equation::burgers::Burgers, burgers::Godunov>::name = "godunov6";


    template class TecnoCombined6<::alsfvm::equation::burgers::Burgers, ScalarEntropyConservativeFlux<equation::burgers::Burgers > >;
    template class TecnoCombined6<::alsfvm::equation::euler::Euler, euler::Tecno1>;
    template class TecnoCombined6<::alsfvm::equation::burgers::Burgers, burgers::Godunov>;
}
}
