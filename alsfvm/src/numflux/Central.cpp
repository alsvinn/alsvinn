#include "alsfvm/numflux/Central.hpp"
#include "alsfvm/equation/equation_list.hpp"
namespace alsfvm { namespace numflux {
    template<class Equation>
    const std::string Central<Equation>::name = "central";

    ALSFVM_EQUATION_INSTANTIATE(Central)
}
}
