#include "alsfvm/equation/euler/Euler.hpp"

namespace alsfvm {
namespace equation {
namespace euler {


template<>
const std::vector<std::string> Euler<3>::conservedVariables = { "rho",
                                             "mx",
                                             "my",
                                             "mz",
                                             "E"
                                         };


template<>
const std::vector<std::string> Euler<3>::primitiveVariables = { "rho",
                                             "ux",
                                             "uy",
                                             "uz",
                                             "p"
                                         };
template<>
const std::vector<std::string> Euler<3>::extraVariables = { "p",
                                             "ux",
                                             "uy",
                                             "uz"
                                         };

template<>
const std::vector<std::string> Euler<2>::conservedVariables = { "rho",
                                             "mx",
                                             "my",
                                             "E"
                                         };


template<>
const std::vector<std::string> Euler<2>::primitiveVariables = { "rho",
                                             "ux",
                                             "uy",
                                             "p"
                                         };

template<>
const std::vector<std::string> Euler<2>::extraVariables = { "p",
                                             "ux",
                                             "uy"
                                         };

template<>
const std::vector<std::string> Euler<1>::conservedVariables = { "rho",
                                             "mx",
                                             "E"
                                         };


template<>
const std::vector<std::string> Euler<1>::primitiveVariables = { "rho",
                                             "ux",
                                             "p"
                                         };


template<>
const std::vector<std::string> Euler<1>::extraVariables = { "p",
                                             "ux"
                                         };



}
}
}
