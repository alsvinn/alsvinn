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

TEST(MatrixTest, MatrixMatrixMultiplication) {
    matrix5 A;
    for (int i = 1; i < 6; ++i) {
        for (int j = 1; j < 6; ++j) {
            A(i - 1, j - 1) = i * 5 + j;
        }
    }
    matrix5 B;
    for (int i = 6; i < 11; ++i) {
        for (int j = 6; j < 11; ++j) {
            B(i - 6, j - 6) = i * 5 + j;
        }
    }


    matrix5 product = A*B;

    // correct answer computed from python:

    std::vector<real> correctAnswer =
    { 1890,  1930,  1970,  2010,  2050,
        3040,  3105,  3170,  3235,  3300,
        4190,  4280,  4370,  4460,  4550,
        5340,  5455,  5570,  5685,  5800,
        6490,  6630,  6770,  6910,  7050 };

    for (int i = 0; i < 5; ++i) {

        for (int j = 0; j < 5; ++j) {
            ASSERT_FLOAT_EQ(correctAnswer[i * 5 + j], product(i, j))
                << "Wrong product value in (" << i << ", " << j << ")" << std::endl
                << "product = " << product << std::endl
                << "A = " << A << std::endl
                << "B = " << B << std::endl;
        }
    }
}