#pragma once

#include <tuple>
#include "alsfvm/equation/burgers/Burgers.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/equation/EquationInformation.hpp"
#include <boost/fusion/algorithm.hpp>

///
/// Defines the list of available equations and an easy way to loop over all
/// of them.
///

namespace alsfvm { namespace equation {



///
/// \brief EquationList is a type list of all equations available.
///
    typedef boost::fusion::vector<EquationInformation<euler::Euler>,
                       EquationInformation<burgers::Burgers> > EquationList;

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



}}

///
/// Macro to instantiate a class for every equation available.
///
#define ALSFVM_EQUATION_INSTANTIATE(X) \
    template class X<::alsfvm::equation::euler::Euler>; \
    template class X<::alsfvm::equation::burgers::Burgers>;
