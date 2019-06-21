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

#include <gtest/gtest.h>
#include "alsfvm/types.hpp"
#include "utils/polyfit.hpp"

using namespace alsfvm;
TEST(Polyfit, LinearPolynomial) {
    // simple test to make sure polyfit works

    size_t N = 1000;
    double a = M_PI;
    double b = std::exp(1);

    std::vector<double> x(N);
    std::vector<double> y(N);

    for (size_t i = 0; i < N; ++i) {
        double xi = double(i) / N;
        x[i] = xi;
        y[i] = a * xi + b;
    }

    auto fit = linearFit(x, y);

    ASSERT_FLOAT_EQ(a, fit[0]);
    ASSERT_FLOAT_EQ(b, fit[1]);
}

TEST(Polyfit, CompareToNumpy) {
    std::vector<double> y = { -2.71871891, -4.06141088, -5.4221225, -6.79844358, -8.17962513,
                            -9.56341751
                        };

    std::vector<double> x = { 4.15888308,  4.85203026,  5.54517744,  6.23832463,  6.93147181,
                            7.62461899
                        };

    ASSERT_FLOAT_EQ(-1.97667591575, linearFit(x, y)[0]);

}
