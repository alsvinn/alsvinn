#pragma once

#include "alsfvm/equation/burgers/Burgers.hpp"
#include "alsfvm/equation/buckleyleverett/BuckleyLeverett.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/equation/cubic/Cubic.hpp"
#include "alsfvm/equation/EquationInformation.hpp"
#include <boost/fusion/algorithm.hpp>

///
/// Defines the list of available equations and an easy way to loop over all
/// of them.
///

namespace alsfvm {
namespace equation {



///
/// \brief EquationList is a type list of all equations available.
///
typedef boost::fusion::vector<EquationInformation<euler::Euler<1>>,
        EquationInformation<euler::Euler<2>>,
        EquationInformation<euler::Euler<3>>,
        EquationInformation<burgers::Burgers>,
        EquationInformation<buckleyleverett::BuckleyLeverett>,
        EquationInformation<cubic::Cubic> > EquationList;

///
/// Loops through each Equation element. Example usage
///
/// \code{.cpp}
/// struct Functor {
///     template<class T>
///     void operator()(const T& t) const {
///         std::cout << T::getName() << std::endl;
///     }
/// };
///
/// for_each_equation(Functor());
/// \endcode
///
template<class Function>
void for_each_equation(const Function& f) {
    EquationList equationList;
    boost::fusion::for_each(equationList, f);
}



}
}

///
/// Macro to instantiate a class for every equation available.
///
#define ALSFVM_EQUATION_INSTANTIATE(X) \
    template class X< ::alsfvm::equation::euler::Euler<1> >; \
    template class X< ::alsfvm::equation::euler::Euler<2> >; \
    template class X< ::alsfvm::equation::euler::Euler<3> >; \
    template class X< ::alsfvm::equation::burgers::Burgers>; \
    template class X< ::alsfvm::equation::buckleyleverett::BuckleyLeverett>; \
    template class X< ::alsfvm::equation::cubic::Cubic>;
