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
    Euler equation(parameters);
	AllVariables input(1, 2, 3, 4, 5, 6, 7, 8, 9);
	// Test for each direction
	{

		ConservedVariables output(1, 1, 1, 1, 1);
        HLL::computeFlux<0>(equation, input, input, output);

		ConservedVariables pointFlux(0, 0, 0, 0, 0);

        equation.computePointFlux<0>(input, pointFlux);

		ASSERT_EQ(pointFlux.E, output.E);
		ASSERT_EQ(pointFlux.m, output.m);
		ASSERT_EQ(pointFlux.rho, output.rho);
	}

	{

		ConservedVariables output(1, 1, 1, 1, 1);
        HLL::computeFlux<1>(equation, input, input, output);
		ConservedVariables pointFlux(0, 0, 0, 0, 0);

        equation.computePointFlux<1>(input, pointFlux);

		ASSERT_EQ(pointFlux.E, output.E);
		ASSERT_EQ(pointFlux.m, output.m);
		ASSERT_EQ(pointFlux.rho, output.rho);
	}

	{
		ConservedVariables output(1, 1, 1, 1, 1);
        HLL::computeFlux<2>(equation, input, input, output);
		ConservedVariables pointFlux(0, 0, 0, 0, 0);
	
        equation.computePointFlux<2>(input, pointFlux);

		ASSERT_EQ(pointFlux.E, output.E);
		ASSERT_EQ(pointFlux.m, output.m);
		ASSERT_EQ(pointFlux.rho, output.rho);
	}
}
