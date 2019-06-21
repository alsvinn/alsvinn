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

#include "gtest/gtest.h"
#include "alsfvm/equation/euler/Euler.hpp"


using namespace alsfvm::equation::euler;
using namespace alsfvm;
TEST(EulerEquationTest, FluxTest) {
    // This test checks that the point flux is correction setup

    // First we check that we get the correct output if everything is one

    AllVariables<3> input(1, rvec3{ 1, 1, 1 }, 1, 1, rvec3{ 1, 1, 1 });

    EulerParameters parameters;
    Euler<3> equation(parameters);
    {
        ConservedVariables<3> output(0, rvec3{ 0, 0, 0 }, 0);

        equation.computePointFlux < 0 >(input, output);

        ASSERT_EQ(output.E, 2);
        ASSERT_EQ(output.m, rvec3(2, 1, 1));
        ASSERT_EQ(output.rho, 1);
    }


    {
        ConservedVariables<3> output(0, rvec3{ 0, 0, 0 }, 0);
        equation.computePointFlux < 1 >(input, output);
        ASSERT_EQ(output.E, 2);
        ASSERT_EQ(output.m, rvec3(1, 2, 1));
        ASSERT_EQ(output.rho, 1);
    }

    {
        ConservedVariables<3> output(0, rvec3{ 0, 0, 0 }, 0);

        equation.computePointFlux < 2 >(input, output);

        ASSERT_EQ(output.E, 2);
        ASSERT_EQ(output.m, rvec3(1, 1, 2));
        ASSERT_EQ(output.rho, 1);
    }
}
