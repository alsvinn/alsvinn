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

TEST(GridTest, CellLengthsTest) {
    rvec3 origin(0, 0, 0);
    rvec3 top(1, 1, 0);

    ivec3 dimensions(30, 30, 1);

    grid::Grid grid(origin, top, dimensions);

    ASSERT_EQ(1. / 30.0, grid.getCellLengths().x);
    ASSERT_EQ(1. / 30.0, grid.getCellLengths().y);
}

TEST(GridTest, MidPointTest2D) {
    rvec3 origin(0, 0, 0);
    rvec3 top(1, 1, 0);

    ivec3 dimensions(30, 30, 1);

    grid::Grid grid(origin, top, dimensions);
    auto midPoints = grid.getCellMidpoints();

    for (size_t y = 0; y < 30; y++) {
        for (size_t x = 0; x < 30; x++) {
            real midpointX = 0 + 1.0 / 30.0 * x + 1.0 / 60.0;
            real midpointY = 0 + 1.0 / 30.0 * y + 1.0 / 60.0;

            const size_t index = y * 30 + x;

            ASSERT_NEAR(midpointX, midPoints[index].x, 1e-8);
            ASSERT_NEAR(midpointY, midPoints[index].y, 1e-8);
        }
    }
}
