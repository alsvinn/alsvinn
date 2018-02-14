#include "alsfvm/numflux/TecnoCombined4.hpp"
#include "alsfvm/numflux/ScalarEntropyConservativeFlux.hpp"
#include "alsfvm/equation/equation_list.hpp"

#include "alsfvm/numflux/euler/Tecno1.hpp"
namespace alsfvm {
namespace numflux {

template<>
const std::string
TecnoCombined4<::alsfvm::equation::burgers::Burgers, ScalarEntropyConservativeFlux<equation::burgers::Burgers > >::name
    = "tecno4";

template<>
const std::string
TecnoCombined4<::alsfvm::equation::euler::Euler<1>, euler::Tecno1<1> >::name =
    "tecno4";

template<>
const std::string
TecnoCombined4<::alsfvm::equation::euler::Euler<2>, euler::Tecno1<2> >::name =
    "tecno4";


template<>
const std::string
TecnoCombined4<::alsfvm::equation::euler::Euler<3>, euler::Tecno1<3> >::name =
    "tecno4";


template<>
const std::string
TecnoCombined4<::alsfvm::equation::burgers::Burgers, burgers::Godunov>::name =
    "godunov4";


template class
TecnoCombined4<::alsfvm::equation::burgers::Burgers, ScalarEntropyConservativeFlux<equation::burgers::Burgers > >;
template class
TecnoCombined4<::alsfvm::equation::euler::Euler<1>, euler::Tecno1<1> >;
template class
TecnoCombined4<::alsfvm::equation::euler::Euler<2>, euler::Tecno1<2> >;
template class
TecnoCombined4<::alsfvm::equation::euler::Euler<3>, euler::Tecno1<3> >;
template class
TecnoCombined4<::alsfvm::equation::burgers::Burgers, burgers::Godunov>;
}
}
