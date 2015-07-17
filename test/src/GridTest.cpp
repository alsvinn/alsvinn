#include "gtest/gtest.h"
#include "alsfvm/grid/Grid.hpp"
using namespace alsfvm;

TEST(GridTest, GetterTest) {
	rvec3 origin(1, 2, 3);
	rvec3 top(10, 10, 10);

	ivec3 dimensions(30, 40, 50);

	grid::Grid grid(origin, top, dimensions);

	ASSERT_EQ(origin, grid.getOrigin());
	ASSERT_EQ(top, grid.getTop());
	ASSERT_EQ(dimensions, grid.getDimensions());
}