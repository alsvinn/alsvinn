#include <gtest/gtest.h>

#include "alsfvm/mpi/cartesian/displacements.hpp"
#include "alsfvm/mpi/cartesian/lengths.hpp"
#include "alsfvm/mpi/cartesian/number_of_segments.hpp"
using namespace alsfvm::mpi::cartesian;

TEST(CartesianDisplacements, NumberOfSegments1D) {
    const int N = 1024;

    int numberOfSegmentsLeft = computeNumberOfSegments(0, 1, {N, 1, 1});
    ASSERT_EQ(1, numberOfSegmentsLeft);

    int numberOfSegmentsRight = computeNumberOfSegments(1, 1, {N, 1, 1});
    ASSERT_EQ(1, numberOfSegmentsRight);
}

TEST(CartesianDisplacements, Lenghts1D) {
    const int N = 1024;
    const int ghostCells = 3;
    auto lengthsLeft = computeLengths(0, 1, {N, 1, 1}, ghostCells);
    ASSERT_EQ(1, lengthsLeft.size());
    ASSERT_EQ(ghostCells, lengthsLeft[0]);

    auto lengthsRight = computeLengths(1, 1, {N, 1, 1}, ghostCells);
    ASSERT_EQ(1, lengthsRight.size());
    ASSERT_EQ(ghostCells, lengthsRight[0]);
}

TEST(CartesianDisplacements, Displacements1DWithoutOffset) {
    const int N = 1024;
    const int ghostCells = 3;
    auto displacementsLeft = computeDisplacements(0, 1, {N, 1, 1}, ghostCells, 0);
    ASSERT_EQ(1, displacementsLeft.size());
    ASSERT_EQ(0, displacementsLeft[0]);

    auto displacementsRight = computeDisplacements(1, 1, {N, 1, 1}, ghostCells, 0);
    ASSERT_EQ(1, displacementsRight.size());
    ASSERT_EQ(N - ghostCells, displacementsRight[0]);
}


TEST(CartesianDisplacements, Displacements1DWithOffset) {
    const int N = 1024;
    const int ghostCells = 3;
    auto displacementsLeft = computeDisplacements(0, 1, {N, 1, 1}, ghostCells,
            ghostCells);
    ASSERT_EQ(1, displacementsLeft.size());
    ASSERT_EQ(ghostCells, displacementsLeft[0]);

    auto displacementsRight = computeDisplacements(1, 1, {N, 1, 1}, ghostCells,
            ghostCells);
    ASSERT_EQ(1, displacementsRight.size());
    ASSERT_EQ(N - ghostCells - ghostCells, displacementsRight[0]);
}







TEST(CartesianDisplacements, NumberOfSegments2D) {
    const int N = 1024;

    int numberOfSegmentsLeft = computeNumberOfSegments(0, 2, {N, N, 1});
    ASSERT_EQ(N, numberOfSegmentsLeft);

    int numberOfSegmentsRight = computeNumberOfSegments(1, 2, {N, N, 1});
    ASSERT_EQ(N, numberOfSegmentsRight);


    int numberOfSegmentBottom = computeNumberOfSegments(2, 2, {N, N, 1});
    ASSERT_EQ(1, numberOfSegmentBottom);

    int numberOfSegmentsTop = computeNumberOfSegments(3, 2, {N, N, 1});
    ASSERT_EQ(1, numberOfSegmentsTop);
}

TEST(CartesianDisplacements, Lenghts2D) {
    const int N = 1024;
    const int ghostCells = 3;
    auto lengthsLeft = computeLengths(0, 2, {N, N, 1}, ghostCells);
    ASSERT_EQ(N, lengthsLeft.size());

    for (int i = 0; i < N; ++i) {
        ASSERT_EQ(ghostCells, lengthsLeft[i]);
    }

    auto lengthsRight = computeLengths(1, 2, {N, N, 1}, ghostCells);
    ASSERT_EQ(N, lengthsRight.size());

    for (int i = 0; i < N; ++i) {
        ASSERT_EQ(ghostCells, lengthsLeft[i]);
    }



    auto lengthsBottom = computeLengths(2, 2, {N, N, 1}, ghostCells);
    ASSERT_EQ(1, lengthsBottom.size());
    ASSERT_EQ(N * ghostCells, lengthsBottom[0]);


    auto lengthsTop = computeLengths(3, 2, {N, N, 1}, ghostCells);
    ASSERT_EQ(1, lengthsTop.size());
    ASSERT_EQ(N * ghostCells, lengthsTop[0]);

}

TEST(CartesianDisplacements, Displacements2DWithoutOffset) {
    const int N = 1024;
    const int ghostCells = 3;
    auto displacementsLeft = computeDisplacements(0, 2, {N, N, 1}, ghostCells, 0);
    ASSERT_EQ(N, displacementsLeft.size());
    ASSERT_EQ(0, displacementsLeft[0]);

    for (int i = 1; i < N; ++i) {
        ASSERT_EQ(displacementsLeft[i], displacementsLeft[i - 1] + N);
    }

    auto displacementsRight = computeDisplacements(1, 2, {N, N, 1}, ghostCells, 0);
    ASSERT_EQ(N, displacementsRight.size());
    ASSERT_EQ(N - ghostCells, displacementsRight[0]);

    for (int i = 1; i < N; ++i) {
        ASSERT_EQ(displacementsRight[i], displacementsRight[i - 1] + N);
    }



    auto displacementsBottom = computeDisplacements(2, 2, {N, N, 1}, ghostCells, 0);
    ASSERT_EQ(1, displacementsBottom.size());
    ASSERT_EQ(0, displacementsBottom[0]);


    auto displacementsTop = computeDisplacements(3, 2, {N, N, 1}, ghostCells, 0);
    ASSERT_EQ(1, displacementsTop.size());
    ASSERT_EQ(N * N - ghostCells * N, displacementsTop[0]);
}


TEST(CartesianDisplacements, Displacements2DWithOffset) {
    const int N = 1024;
    const int ghostCells = 3;
    auto displacementsLeft = computeDisplacements(0, 2, {N, N, 1}, ghostCells,
            ghostCells);
    ASSERT_EQ(N, displacementsLeft.size());
    ASSERT_EQ(ghostCells, displacementsLeft[0]);

    for (int i = 1; i < N; ++i) {
        ASSERT_EQ(displacementsLeft[i], displacementsLeft[i - 1] + N);
    }

    auto displacementsRight = computeDisplacements(1, 2, {N, N, 1}, ghostCells,
            ghostCells);
    ASSERT_EQ(N, displacementsRight.size());
    ASSERT_EQ(N - 2 * ghostCells, displacementsRight[0]);

    for (int i = 1; i < N; ++i) {
        ASSERT_EQ(displacementsRight[i], displacementsRight[i - 1] + N);
    }



    auto displacementsBottom = computeDisplacements(2, 2, {N, N, 1}, ghostCells,
            ghostCells);
    ASSERT_EQ(1, displacementsBottom.size());
    ASSERT_EQ(ghostCells * N, displacementsBottom[0]);


    auto displacementsTop = computeDisplacements(3, 2, {N, N, 1}, ghostCells,
            ghostCells);
    ASSERT_EQ(1, displacementsTop.size());
    ASSERT_EQ(N * N - 2 * ghostCells * N, displacementsTop[0]);
}

