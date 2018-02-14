#include "alsfvm/numflux/ScalarEntropyConservativeFlux.hpp"
#include "alsfvm/equation/equation_list.hpp"
namespace alsfvm {
namespace numflux {
template<class Equation>
const std::string ScalarEntropyConservativeFlux<Equation>::name = "tecno1";

ALSFVM_EQUATION_INSTANTIATE(ScalarEntropyConservativeFlux)
}
}
