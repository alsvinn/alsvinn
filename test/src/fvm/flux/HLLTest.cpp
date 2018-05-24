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
#include "alsfvm/numflux/euler/HLL.hpp"

using namespace alsfvm::equation::euler;
using namespace alsfvm::numflux::euler;
using namespace alsfvm;

TEST(HLLTest, ConsistencyTest) {
    // Here we test that if left==right,
    // then the numerical flux just returns the point flux


    EulerParameters parameters;
    Euler<3> equation(parameters);
    AllVariables<3> input(1, rvec3{ 2, 3, 4 }, 5, 6, rvec3{ 7, 8, 9 });
    // Test for each direction
    {

        ConservedVariables<3> output(1, rvec3{ 1, 1, 1 }, 1);
        HLL<3>::computeFlux<0>(equation, input, input, output);

        ConservedVariables<3> pointFlux(0, rvec3{ 0, 0, 0 }, 0);

        equation.computePointFlux<0>(input, pointFlux);

        ASSERT_EQ(pointFlux.E, output.E);
        ASSERT_EQ(pointFlux.m, output.m);
        ASSERT_EQ(pointFlux.rho, output.rho);
    }

    {

        ConservedVariables<3> output(1, rvec3{ 1, 1, 1 }, 1);
        HLL<3>::computeFlux<1>(equation, input, input, output);
        ConservedVariables<3> pointFlux(0, rvec3{ 0, 0, 0 }, 0);

        equation.computePointFlux<1>(input, pointFlux);

        ASSERT_EQ(pointFlux.E, output.E);
        ASSERT_EQ(pointFlux.m, output.m);
        ASSERT_EQ(pointFlux.rho, output.rho);
    }

    {
        ConservedVariables<3> output(1, rvec3{ 1, 1, 1 }, 1);
        HLL<3>::computeFlux<2>(equation, input, input, output);
        ConservedVariables<3> pointFlux(0, rvec3{ 0, 0, 0 }, 0);

        equation.computePointFlux<2>(input, pointFlux);

        ASSERT_EQ(pointFlux.E, output.E);
        ASSERT_EQ(pointFlux.m, output.m);
        ASSERT_EQ(pointFlux.rho, output.rho);
    }
}
