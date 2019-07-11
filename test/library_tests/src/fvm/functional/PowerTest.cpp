/* Copyright (c) 2019 ETH Zurich, Kjetil Olsen Lye
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
#include "alsutils/math/FastPower.hpp"
#include "alsutils/math/PowPower.hpp"

TEST(PowerTest, TestPowPower) {
    for (int p = 1; p < 10; ++p) {
        // I would have tested for unlimited power here, but the senate would not
        // approve.
        ASSERT_EQ(alsutils::math::PowPower::power(2, p), std::pow(2, p));
    }
}

TEST(PowerTest, TestFastPower) {
    ASSERT_EQ(alsutils::math::FastPower<1>::power(2, 1), std::pow(2, 1));


    ASSERT_EQ(alsutils::math::FastPower<2>::power(2, 2), std::pow(2, 2));


    ASSERT_EQ(alsutils::math::FastPower<3>::power(2, 3), std::pow(2, 3));


    ASSERT_EQ(alsutils::math::FastPower<4>::power(2, 4), std::pow(2, 4));


    ASSERT_EQ(alsutils::math::FastPower<5>::power(2, 5), std::pow(2, 5));
}
