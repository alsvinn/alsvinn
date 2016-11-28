#include <gtest/gtest.h>
#include "alsfvm/types.hpp"
using namespace alsfvm;
TEST(MatrixTests, Identity) {
    auto identity = matrix5::identity();

    rvec5 vec{ 1,2,3,4,5 };

    ASSERT_EQ(vec, identity*vec);
}

TEST(MatrixTests, DenseMatrixTest) {
    matrix5 matrix;

    // Fill it with a recognizable pattern
    for (int i = 1; i <= 5; ++i) {
        for (int j = 1; j <= 5; ++j) {
            matrix(i - 1, j - 1) = i * 5 + j;
        }
    }

    // first test with basis vectors
    for (int i = 1; i <= 5; ++i) {
        rvec5 vec;
        vec[i-1] = 1;
        auto multiplicationResult = matrix * vec;

        for (int j = 1; j <= 5; j++) {
            ASSERT_EQ(j * 5 + i, multiplicationResult[j-1]);
        }
    }

    rvec5 vec{ 1,2,3,4,5 };

    auto multiplicationResult = matrix * vec;
    for (int i = 1; i <= 5; ++i) {
        real expectedResult = 0;
        for (int j = 1; j <= 5; ++j) {
            expectedResult += (5 * i + j) * j;
        }

        ASSERT_EQ(expectedResult, multiplicationResult[i - 1]);
    }
}