#include <gtest/gtest.h>
#include "alsfvm/equation/euler/Euler.hpp"
using namespace alsfvm;
using namespace alsfvm::equation;
using namespace alsfvm::equation::euler;

TEST(TecnoTest, TestTecnoVariables) {
    EulerParameters parameters;
    Euler equation(parameters);

    ConservedVariables conserved(10,0,0,0,14);

    auto primitive = equation.computePrimitiveVariables(conserved);

    auto tecnoVariables = equation.computeTecnoVariables(conserved);

    ASSERT_FLOAT_EQ(sqrt(primitive.rho/primitive.p), tecnoVariables.z[0]);
    ASSERT_FLOAT_EQ(0, tecnoVariables.z[1]);
    ASSERT_FLOAT_EQ(0, tecnoVariables.z[2]);
    ASSERT_FLOAT_EQ(0, tecnoVariables.z[3]);
    ASSERT_FLOAT_EQ(sqrt(primitive.rho * primitive.p), tecnoVariables.z[4]);

}
