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
#include "alsfvm/functional/FunctionalFactory.hpp"
#include "alsfvm/volume/make_volume.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/equation/euler/Euler.hpp"

using namespace alsfvm;
TEST(LegendreTest, CreationTest) {

    alsfvm::functional::FunctionalFactory factory;
    alsfvm::functional::Functional::Parameters parameters({{"minValue", "-1"},
        {"maxValue", "1"},
        {"degree_k", "1"},
        {"degree_n", "1"},
        {"degree_m", "1"}});

    auto functional = factory.makeFunctional("cpu", "legendre", parameters);

    grid::Grid grid({0, 0, 0}, {1, 1, 1}, {40, 42, 43});
    ASSERT_EQ(ivec3(1, 1, 1), functional->getFunctionalSize(grid));
}

TEST(LegendreTest, ConstantTest) {
    alsfvm::functional::FunctionalFactory factory;
    alsfvm::functional::Functional::Parameters parameters({{"minValue", "-1"},
        {"maxValue", "1"},
        {"degree_k", "0"},
        {"degree_n", "0"},
        {"degree_m", "0"},
    });
    auto functional = factory.makeFunctional("cpu", "legendre", parameters);

    ivec3 size = {3, 2, 1};
    auto conservedIn = alsfvm::volume::makeConservedVolume("cpu", "euler2", size,
            2);

    grid::Grid grid({0, 0, 0}, {1, 1, 1}, size);
    volume::fill_volume<equation::euler::ConservedVariables<2> >(*conservedIn,
    grid, [](real x, real y, real z, equation::euler::ConservedVariables<2>& out) {
        out.rho = sin(2 * M_PI * x * y);
        out.m.x = sin(2 * M_PI * x * y);
        out.m.y = sin(2 * M_PI * x * y);
        out.E = sin(2 * M_PI * x * y);
    });

    auto conservedOut = alsfvm::volume::makeConservedVolume("cpu", "euler2", {1, 1, 1},
            2);
    conservedOut->makeZero();
    auto extraIn = alsfvm::volume::makeExtraVolume("cpu", "euler2", size, 2);

    volume::fill_volume<equation::euler::ExtraVariables<2> >(*extraIn,
    grid, [](real x, real y, real z, equation::euler::ExtraVariables<2>& out) {
        out.p = sin(2 * M_PI * x * y);
        out.u.x = sin(2 * M_PI * x * y);
        out.u.y = sin(2 * M_PI * x * y);
    });

    auto extraOut = alsfvm::volume::makeExtraVolume("cpu", "euler2", {1, 1, 1}, 0);
    extraOut->makeZero();
    functional->operator ()(*conservedOut, *conservedIn, 0.5,
        grid);

    for (size_t var = 0; var < conservedOut->getNumberOfVariables(); ++var) {
        ASSERT_DOUBLE_EQ(0.5, conservedOut->getScalarMemoryArea(var)->getPointer()[0]);
    }

}
