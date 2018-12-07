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

#include <gtest/gtest.h>
#include "alsfvm/equation/equation_list.hpp"
#include <iostream>

using namespace alsfvm::equation;

struct NamesFoundFunctor {
    NamesFoundFunctor(std::vector<std::string>& namesFound)
        : namesFound(namesFound) {

    }
    template<class T>
    void operator()(T& t) const {
        namesFound.push_back(T::getName());
    }

    std::vector<std::string>& namesFound;
};


TEST(EquationListTest, NameTest) {
    ASSERT_EQ("euler3", EquationInformation<euler::Euler<3>>::getName());
    ASSERT_EQ("euler2", EquationInformation<euler::Euler<2>>::getName());
    ASSERT_EQ("euler1", EquationInformation<euler::Euler<1>>::getName());
    ASSERT_EQ("burgers", EquationInformation<burgers::Burgers>::getName());
}

TEST(EquationListTest, CheckNames) {
    std::vector<std::string> namesFound;
    NamesFoundFunctor namesFoundFunctor(namesFound);
    alsfvm::equation::for_each_equation(namesFoundFunctor);

    ASSERT_TRUE(std::find(namesFoundFunctor.namesFound.begin(),
            namesFoundFunctor.namesFound.end(), "euler1")
        != namesFoundFunctor.namesFound.end());

    ASSERT_TRUE(std::find(namesFoundFunctor.namesFound.begin(),
            namesFoundFunctor.namesFound.end(), "euler2")
        != namesFoundFunctor.namesFound.end());


    ASSERT_TRUE(std::find(namesFoundFunctor.namesFound.begin(),
            namesFoundFunctor.namesFound.end(), "euler3")
        != namesFoundFunctor.namesFound.end());


    ASSERT_TRUE(std::find(namesFoundFunctor.namesFound.begin(),
            namesFoundFunctor.namesFound.end(), "burgers")
        != namesFoundFunctor.namesFound.end());

}

