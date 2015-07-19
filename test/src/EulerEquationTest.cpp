#include "gtest/gtest.h"
#include "alsfvm/equation/euler/Euler.hpp"

using namespace alsfvm::equation::euler;
using namespace alsfvm;
TEST(EulerEquationTest, FluxTest) {
	// This test checks that the point flux is correction setup

	// First we check that we get the correct output if everything is one

	AllVariables input(1, 1, 1, 1, 1, 1, 1, 1, 1);

	{
		ConservedVariables output(0, 0, 0, 0, 0);

		Euler::computePointFlux < 0 >(input, output);

		ASSERT_EQ(output.E, 2);
		ASSERT_EQ(output.m, rvec3(2, 1, 1));
		ASSERT_EQ(output.rho, 1);
	}

		
	{
		ConservedVariables output(0, 0, 0, 0, 0);
		Euler::computePointFlux < 1 >(input, output);
		ASSERT_EQ(output.E, 2);
		ASSERT_EQ(output.m, rvec3(1, 2, 1));
		ASSERT_EQ(output.rho, 1);
	}

	{
		ConservedVariables output(0, 0, 0, 0, 0);

		Euler::computePointFlux < 2 >(input, output);

		ASSERT_EQ(output.E, 2);
		ASSERT_EQ(output.m, rvec3(1, 1, 2));
		ASSERT_EQ(output.rho, 1);
	}
}