#include <gtest/gtest.h>
#include "alsutils/math/legendre.hpp"
#include <boost/math/special_functions/legendre.hpp>

// Simly test if our legendre polynomials agree with boost's version
TEST(LegendreTest, AgainstBoost) {

    // We discretize the interval -1 to 1 into N + 1pieces
    const int N = 10;

    const int dx = 1.0 / N;
    const int maxDegree = 11;

    for (int degree = 0; degree <= maxDegree; ++ degree) {
        for (int i = 0; i < N+1; ++i) {
            const double x = i * dx;

            ASSERT_FLOAT_EQ(boost::math::legendre_p(degree, x),
                            alsutils::math::legendre_p(degree, x))
                    << "\nFailed with {\n\tdegree = " << degree << "\n"
                    << "\tx = " << x << "\n}\n\n";
        }
    }
}
