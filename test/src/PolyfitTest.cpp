#include <gtest/gtest.h>
#include "alsfvm/types.hpp"
#include "utils/polyfit.hpp"

using namespace alsfvm;
TEST(Polyfit, LinearPolynomial) {
    // simple test to make sure polyfit works

    size_t N = 1000;
    real a = M_PI;
    real b = std::exp(1);

    std::vector<real> x(N);
    std::vector<real> y(N);

    for (size_t i = 0; i < N; ++i) {
        real xi = real(i) / N;
        x[i] = xi;
        y[i] = a*xi + b;
    }

    auto fit = linearFit(x, y);

    ASSERT_FLOAT_EQ(a, fit[0]);
    ASSERT_FLOAT_EQ(b, fit[1]);
}