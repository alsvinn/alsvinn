#include "alsfvm/numflux/Rusanov.hpp"
#include "alsfvm/equation/equation_list.hpp"
namespace alsfvm {
namespace numflux {
template<class Equation>
const std::string Rusanov<Equation>::name = "rusanov";

template class Rusanov<equation::burgers::Burgers>;
template class Rusanov<equation::buckleyleverett::BuckleyLeverett>;
template class Rusanov<equation::cubic::Cubic>;

}
}
