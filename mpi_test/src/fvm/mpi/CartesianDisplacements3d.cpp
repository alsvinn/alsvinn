#include <gtest/gtest.h>

#include "alsfvm/mpi/cartesian/displacements.hpp"
#include "alsfvm/mpi/cartesian/lengths.hpp"
#include "alsfvm/mpi/cartesian/number_of_segments.hpp"
using namespace alsfvm::mpi::cartesian;




TEST(CartesianDisplacements, NumberOfSegments3D) {
    const int N = 1024;

    int numberOfSegmentsLeft = computeNumberOfSegments(0, 3, {N, N, N});
    ASSERT_EQ(N * N, numberOfSegmentsLeft);

    int numberOfSegmentsRight = computeNumberOfSegments(1, 3, {N, N, N});
    ASSERT_EQ(N * N, numberOfSegmentsRight);


    int numberOfSegmentBottom = computeNumberOfSegments(2, 3, {N, N, N});
    ASSERT_EQ(N, numberOfSegmentBottom);

    int numberOfSegmentsTop = computeNumberOfSegments(3, 3, {N, N, N});
    ASSERT_EQ(N, numberOfSegmentsTop);



    int numberOfSegmentFront = computeNumberOfSegments(4, 3, {N, N, N});
    ASSERT_EQ(1, numberOfSegmentFront);

    int numberOfSegmentsBack = computeNumberOfSegments(5, 3, {N, N, N});
    ASSERT_EQ(1, numberOfSegmentsBack);
}

TEST(CartesianDisplacements, Lenghts3D) {
    const int N = 1024;
    const int ghostCells = 3;
    auto lengthsLeft = computeLengths(0, 3, {N, N, N}, ghostCells);
    ASSERT_EQ(N * N, lengthsLeft.size());

    for (int i = 0; i < N * N; ++i) {
        ASSERT_EQ(ghostCells, lengthsLeft[i]);
    }

    auto lengthsRight = computeLengths(1, 3, {N, N, N}, ghostCells);
    ASSERT_EQ(N * N, lengthsRight.size());

    for (int i = 0; i < N * N; ++i) {
        ASSERT_EQ(ghostCells, lengthsLeft[i]);
    }



    auto lengthsBottom = computeLengths(2, 3, {N, N, N}, ghostCells);
    ASSERT_EQ(N, lengthsBottom.size());

    for (int i = 0; i < N; ++i) {
        ASSERT_EQ(ghostCells * N, lengthsBottom[i]);
    }

    auto lengthsTop = computeLengths(3, 3, {N, N, N}, ghostCells);
    ASSERT_EQ(N, lengthsTop.size());

    for (int i = 0; i < N; ++i) {
        ASSERT_EQ(ghostCells * N, lengthsTop[i]);
    }


    auto lengthsFront = computeLengths(4, 3, {N, N, N}, ghostCells);
    ASSERT_EQ(1, lengthsFront.size());
    ASSERT_EQ(N * N * ghostCells, lengthsFront[0]);


    auto lengthsBack = computeLengths(5, 3, {N, N, N}, ghostCells);
    ASSERT_EQ(1, lengthsBack.size());
    ASSERT_EQ(N * N * ghostCells, lengthsBack[0]);

}

TEST(CartesianDisplacements, Displacements3DWithoutOffset) {
    const int N = 1024;
    const int ghostCells = 3;
    auto displacementsLeft = computeDisplacements(0, 3, {N, N, N}, ghostCells, 0);
    ASSERT_EQ(N * N, displacementsLeft.size());
    ASSERT_EQ(0, displacementsLeft[0]);

    for (int i = 1; i < N * N; ++i) {
        ASSERT_EQ(displacementsLeft[i], displacementsLeft[i - 1] + N);
    }

    auto displacementsRight = computeDisplacements(1, 3, {N, N, N}, ghostCells, 0);
    ASSERT_EQ(N * N, displacementsRight.size());
    ASSERT_EQ(N - ghostCells, displacementsRight[0]);

    for (int i = 1; i < N * N; ++i) {
        ASSERT_EQ(displacementsRight[i], displacementsRight[i - 1] + N);
    }



    auto displacementsBottom = computeDisplacements(2, 3, {N, N, N}, ghostCells, 0);
    ASSERT_EQ(N, displacementsBottom.size());
    ASSERT_EQ(0, displacementsBottom[0]);

    for (int i = 1; i < N; ++i) {
        ASSERT_EQ(displacementsBottom[i], displacementsBottom[i - 1] + N * N);
    }


    auto displacementsTop = computeDisplacements(3, 3, {N, N, N}, ghostCells, 0);
    ASSERT_EQ(N, displacementsTop.size());
    ASSERT_EQ(N * N - ghostCells * N, displacementsTop[0]);

    for (int i = 1; i < N; ++i) {
        ASSERT_EQ(displacementsTop[i], displacementsTop[i - 1] + N * N);
    }





    auto displacementsFront = computeDisplacements(4, 3, {N, N, N}, ghostCells, 0);
    ASSERT_EQ(1, displacementsFront.size());
    ASSERT_EQ(0, displacementsFront[0]);



    auto displacementsBack = computeDisplacements(5, 3, {N, N, N}, ghostCells, 0);
    ASSERT_EQ(1, displacementsBack.size());
    ASSERT_EQ(N * N * N - ghostCells * N * N, displacementsBack[0]);
}


TEST(CartesianDisplacements, Displacements3DWithOffset) {
    const int N = 1024;
    const int ghostCells = 3;
    auto displacementsLeft = computeDisplacements(0, 3, {N, N, N}, ghostCells,
            ghostCells);
    ASSERT_EQ(N * N, displacementsLeft.size());
    ASSERT_EQ(ghostCells, displacementsLeft[0]);

    for (int i = 1; i < N * N; ++i) {
        ASSERT_EQ(displacementsLeft[i], displacementsLeft[i - 1] + N);
    }

    auto displacementsRight = computeDisplacements(1, 3, {N, N, N}, ghostCells,
            ghostCells);
    ASSERT_EQ(N * N, displacementsRight.size());
    ASSERT_EQ(N - 2 * ghostCells, displacementsRight[0]);

    for (int i = 1; i < N * N; ++i) {
        ASSERT_EQ(displacementsRight[i], displacementsRight[i - 1] + N);
    }



    auto displacementsBottom = computeDisplacements(2, 3, {N, N, N}, ghostCells,
            ghostCells);
    ASSERT_EQ(N, displacementsBottom.size());
    ASSERT_EQ(N * ghostCells, displacementsBottom[0]);

    for (int i = 1; i < N; ++i) {
        ASSERT_EQ(displacementsBottom[i], displacementsBottom[i - 1] + N * N);
    }


    auto displacementsTop = computeDisplacements(3, 3, {N, N, N}, ghostCells,
            ghostCells);
    ASSERT_EQ(N, displacementsTop.size());
    ASSERT_EQ(N * N - 2 * ghostCells * N, displacementsTop[0]);

    for (int i = 1; i < N; ++i) {
        ASSERT_EQ(displacementsTop[i], displacementsTop[i - 1] + N * N);
    }





    auto displacementsFront = computeDisplacements(4, 3, {N, N, N}, ghostCells,
            ghostCells);
    ASSERT_EQ(1, displacementsFront.size());
    ASSERT_EQ(ghostCells * N * N, displacementsFront[0]);



    auto displacementsBack = computeDisplacements(5, 3, {N, N, N}, ghostCells,
            ghostCells);
    ASSERT_EQ(1, displacementsBack.size());
    ASSERT_EQ(N * N * N - 2 * ghostCells * N * N, displacementsBack[0]);
}

