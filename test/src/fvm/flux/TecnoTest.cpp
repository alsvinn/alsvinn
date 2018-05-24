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
#include "alsfvm/equation/euler/Euler.hpp"
using namespace alsfvm;
using namespace alsfvm::equation;
using namespace alsfvm::equation::euler;

TEST(TecnoTest, TestTecnoVariables) {
    EulerParameters parameters;
    Euler<3> equation(parameters);

    ConservedVariables<3> conserved(10, rvec3{ 0, 0, 0 }, 14);

    auto primitive = equation.computePrimitiveVariables(conserved);

    auto tecnoVariables = equation.computeTecnoVariables(conserved);

    ASSERT_FLOAT_EQ(sqrt(primitive.rho / primitive.p), tecnoVariables.z[0]);
    ASSERT_FLOAT_EQ(0, tecnoVariables.z[1]);
    ASSERT_FLOAT_EQ(0, tecnoVariables.z[2]);
    ASSERT_FLOAT_EQ(0, tecnoVariables.z[3]);
    ASSERT_FLOAT_EQ(sqrt(primitive.rho * primitive.p), tecnoVariables.z[4]);

}


