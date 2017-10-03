#include <gtest/gtest.h>

#include "alsfvm/mpi/cartesian/displacements.hpp"
#include "alsfvm/mpi/cartesian/lengths.hpp"
#include "alsfvm/mpi/cartesian/number_of_segments.hpp"
using namespace alsfvm::mpi::cartesian;

TEST(CartesianDisplacements, NumberOfSegments1D) {
    const int N = 1024;

    int numberOfSegmentsLeft = computeNumberOfSegments(0, 1, {N,1,1});
    ASSERT_EQ(1, numberOfSegmentsLeft);

    int numberOfSegmentsRight= computeNumberOfSegments(1, 1, {N,1,1});
    ASSERT_EQ(1, numberOfSegmentsRight);
}

TEST(CartesianDisplacements, Lenghts1D) {
    const int N = 1024;
    const int ghostCells = 3;
    auto lengthsLeft = computeLengths(0, 1, {N,1,1}, ghostCells);
    ASSERT_EQ(1, lengthsLeft.size());
    ASSERT_EQ(ghostCells, lengthsLeft[0]);

    auto lengthsRight = computeLengths(1, 1, {N,1,1}, ghostCells);
    ASSERT_EQ(1, lengthsRight.size());
    ASSERT_EQ(ghostCells, lengthsRight[0]);
}

TEST(CartesianDisplacements, Displacements1DWithoutOffset) {
    const int N = 1024;
    const int ghostCells = 3;
    auto displacementsLeft = computeDisplacements(0, 1, {N,1,1}, ghostCells, 0);
    ASSERT_EQ(1, displacementsLeft.size());
    ASSERT_EQ(0, displacementsLeft[0]);

    auto displacementsRight = computeDisplacements(1, 1, {N,1,1}, ghostCells, 0);
    ASSERT_EQ(1, displacementsRight.size());
    ASSERT_EQ(N-ghostCells, displacementsRight[0]);
}


TEST(CartesianDisplacements, Displacements1DWithOffset) {
    const int N = 1024;
    const int ghostCells = 3;
    auto displacementsLeft = computeDisplacements(0, 1, {N,1,1}, ghostCells, ghostCells);
    ASSERT_EQ(1, displacementsLeft.size());
    ASSERT_EQ(ghostCells, displacementsLeft[0]);

    auto displacementsRight = computeDisplacements(1, 1, {N,1,1}, ghostCells, ghostCells);
    ASSERT_EQ(1, displacementsRight.size());
    ASSERT_EQ(N-ghostCells-ghostCells, displacementsRight[0]);
}
