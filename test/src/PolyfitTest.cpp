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

TEST(Polyfit, CompareToNumpy) {
    std::vector<real> y = { -2.71871891, -4.06141088, -5.4221225 , -6.79844358, -8.17962513,
        -9.56341751 };

    std::vector<real> x = { 4.15888308,  4.85203026,  5.54517744,  6.23832463,  6.93147181,
        7.62461899 };

    ASSERT_FLOAT_EQ(-1.97667591575, linearFit(x, y)[0]);

}