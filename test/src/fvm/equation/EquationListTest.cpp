#include <gtest/gtest.h>
#include "alsfvm/equation/equation_list.hpp"
#include <iostream>

using namespace alsfvm::equation;

struct NamesFoundFunctor {
    template<class T>
    void operator()(T& t) const {
        namesFound.push_back(T::getName());
    }

    mutable std::vector<std::string> namesFound;
};


TEST(EquationListTest, NameTest) {
    ASSERT_EQ("euler3", EquationInformation<euler::Euler<3>>::getName());
    ASSERT_EQ("euler2", EquationInformation<euler::Euler<2>>::getName());
    ASSERT_EQ("euler1", EquationInformation<euler::Euler<1>>::getName());
    ASSERT_EQ("burgers", EquationInformation<burgers::Burgers>::getName());
}

TEST(EquationListTest, CheckNames) {

    NamesFoundFunctor namesFoundFunctor;
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

