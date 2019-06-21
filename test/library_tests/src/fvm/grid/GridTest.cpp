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
#include "alsfvm/grid/Grid.hpp"
#define TOLERANCE (std::is_same<alsfvm::real, float>::value ? 1e-7 : 1e-8)
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

    ASSERT_EQ(real(1. / 30.0), grid.getCellLengths().x);
    ASSERT_EQ(real(1. / 30.0), grid.getCellLengths().y);
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

            ASSERT_NEAR(midpointX, midPoints[index].x, TOLERANCE);
            ASSERT_NEAR(midpointY, midPoints[index].y, TOLERANCE);
        }
    }
}


TEST(GridTest, MidPointTest3D) {
    rvec3 origin(0, 0, 0);
    rvec3 top(1, 1, 1);

    const int N = 30;
    ivec3 dimensions(N, N, N);

    grid::Grid grid(origin, top, dimensions);
    auto midPoints = grid.getCellMidpoints();

    for (size_t z = 0; z < N; z++) {
        for (size_t y = 0; y < N; y++) {
            for (size_t x = 0; x < N; x++) {
                real midpointX = 0 + 1.0 / N * x + 0.5 / N;
                real midpointY = 0 + 1.0 / N * y + 0.5 / N;
                real midpointZ = 0 + 1.0 / N * z + 0.5 / N;

                const size_t index = z * N * N + y * N  + x;

                ASSERT_NEAR(midpointX, midPoints[index].x, TOLERANCE);
                ASSERT_NEAR(midpointY, midPoints[index].y, TOLERANCE);
                ASSERT_NEAR(midpointZ, midPoints[index].z, TOLERANCE);
            }
        }
    }
}
