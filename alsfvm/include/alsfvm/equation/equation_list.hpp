/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "alsfvm/equation/burgers/Burgers.hpp"
#include "alsfvm/equation/buckleyleverett/BuckleyLeverett.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/equation/cubic/Cubic.hpp"
#include "alsfvm/equation/EquationInformation.hpp"
#include "alsutils/fusion_without_warning.hpp"

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
