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
