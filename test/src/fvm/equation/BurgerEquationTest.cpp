#include <gtest/gtest.h>
#include "alsfvm/types.hpp"
#include "alsfvm/equation/burgers/Burgers.hpp"

using namespace alsfvm::equation::burgers;
using namespace alsfvm::equation;
using namespace alsfvm;
TEST(BurgerEquationTest, FluxTest) {

    const real u = 42.42;
    AllVariables allVariables(u);

    ConservedVariables F;

    EquationParameters parameters;
    Burgers burgersEquation(parameters);

    burgersEquation.computePointFlux<0>(allVariables, F);

    ASSERT_FLOAT_EQ(F.u, u * u / 2);

}
