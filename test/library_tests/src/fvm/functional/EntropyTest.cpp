/* Copyright (c) 2020 ETH Zurich, Kjetil Olsen Lye
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
#include "alsutils/config.hpp"

using namespace alsfvm;
TEST(EntropyTest, CreationTest) {

    alsfvm::functional::FunctionalFactory factory;
    alsfvm::functional::Functional::Parameters parameters({{"gamma", "1.4"}, {"something", "something"}});

    auto functional = factory.makeFunctional("cpu", "log_entropy", parameters);

    grid::Grid grid({0, 0, 0}, {1, 1, 1}, {40, 42, 43});
    ASSERT_EQ(ivec3(1, 1, 1), functional->getFunctionalSize(grid));
}

TEST(EntropyTest, ResultIsZeroTest) {
    alsfvm::functional::FunctionalFactory factory;
    alsfvm::functional::Functional::Parameters parameters({{"gamma", "1.4"}, {"something", "something"}});

    auto functional = factory.makeFunctional("cpu", "log_entropy", parameters);

    ivec3 size = {3, 2, 1};
    auto conservedIn = alsfvm::volume::makeConservedVolume("cpu", "euler2", size,
            2);

    grid::Grid grid({0, 0, 0}, {1, 1, 1}, size);
    volume::fill_volume<equation::euler::ConservedVariables<2> >(*conservedIn,
    grid, [](real x, real y, real z, equation::euler::ConservedVariables<2>& out) {
        out.rho = exp(1.0);
        out.m.x = sin(2 * M_PI * x * y);
        out.m.y = sin(2 * M_PI * x * y);
        out.E = exp(1.4) / 0.4 + .5 * out.m.dot(out.m) / out.rho;
    });

    auto conservedOut = alsfvm::volume::makeConservedVolume("cpu", "euler2", {1, 1, 1},
            2);
    conservedOut->makeZero();
    auto extraIn = alsfvm::volume::makeExtraVolume("cpu", "euler2", size, 2);


    auto extraOut = alsfvm::volume::makeExtraVolume("cpu", "euler2", {1, 1, 1}, 0);
    extraOut->makeZero();
    functional->operator ()(*conservedOut, *conservedIn, 0.5,
        grid);


    ASSERT_NEAR(0.0, conservedOut->getScalarMemoryArea("E")->getPointer()[0],
        1e-8);

}

TEST(EntropyTest, ResultIsOneTest) {
    alsfvm::functional::FunctionalFactory factory;
    alsfvm::functional::Functional::Parameters parameters({{"gamma", "1.4"}, {"something", "something"}});

    auto functional = factory.makeFunctional("cpu", "log_entropy", parameters);

    ivec3 size = {2, 2, 1};
    auto conservedIn = alsfvm::volume::makeConservedVolume("cpu", "euler2", size,
            2);

    grid::Grid grid({0, 0, 0}, {1, 1, 1}, size);
    volume::fill_volume<equation::euler::ConservedVariables<2> >(*conservedIn,
    grid, [](real x, real y, real z, equation::euler::ConservedVariables<2>& out) {
        out.rho = exp(1.0);
        out.m.x = sin(2 * M_PI * x * y);
        out.m.y = sin(2 * M_PI * x * y);
        out.E = exp(1.4 + 0.4 / exp(1)) / 0.4 + .5 * out.m.dot(out.m) / out.rho;
    });

    auto conservedOut = alsfvm::volume::makeConservedVolume("cpu", "euler2", {1, 1, 1},
            2);
    conservedOut->makeZero();
    auto extraIn = alsfvm::volume::makeExtraVolume("cpu", "euler2", size, 2);


    auto extraOut = alsfvm::volume::makeExtraVolume("cpu", "euler2", {1, 1, 1}, 0);
    extraOut->makeZero();
    functional->operator ()(*conservedOut, *conservedIn, -1.0,
        grid);


    ASSERT_NEAR(1.0, conservedOut->getScalarMemoryArea("E")->getPointer()[0], 1e-8);

}

#ifdef ALSVINN_HAVE_CUDA


TEST(EntropyTest, CUDACreationTest) {

    alsfvm::functional::FunctionalFactory factory;
    alsfvm::functional::Functional::Parameters parameters({{"gamma", "1.4"}, {"something", "something"}});

    auto functional = factory.makeFunctional("cuda", "log_entropy", parameters);

    grid::Grid grid({0, 0, 0}, {1, 1, 1}, {40, 42, 43});
    ASSERT_EQ(ivec3(1, 1, 1), functional->getFunctionalSize(grid));
}

TEST(EntropyTest, CUDAResultIsZeroTest) {
    alsfvm::functional::FunctionalFactory factory;
    alsfvm::functional::Functional::Parameters parameters({{"gamma", "1.4"}, {"something", "something"}});

    auto functional = factory.makeFunctional("cuda", "log_entropy", parameters);

    ivec3 size = {3, 2, 1};
    auto conservedIn = alsfvm::volume::makeConservedVolume("cpu", "euler2", size,
            2);
    auto conservedInCUDA = alsfvm::volume::makeConservedVolume("cuda", "euler2",
            size,
            2);

    grid::Grid grid({0, 0, 0}, {1, 1, 1}, size);
    volume::fill_volume<equation::euler::ConservedVariables<2> >(*conservedIn,
    grid, [](real x, real y, real z, equation::euler::ConservedVariables<2>& out) {
        out.rho = exp(1.0);
        out.m.x = sin(2 * M_PI * x * y);
        out.m.y = sin(2 * M_PI * x * y);
        out.E = exp(1.4) / 0.4 + .5 * out.m.dot(out.m) / out.rho;
    });

    conservedIn->copyTo(*conservedInCUDA);

    auto conservedOutCUDA = alsfvm::volume::makeConservedVolume("cuda", "euler2", {1, 1, 1},
            0);
    auto conservedOut = alsfvm::volume::makeConservedVolume("cpu", "euler2", {1, 1, 1},
            0);

    conservedOutCUDA->makeZero();

    functional->operator ()(*conservedOutCUDA, *conservedInCUDA, 0.5,
        grid);

    conservedOutCUDA->copyTo(*conservedOut);

    ASSERT_NEAR(0.0, conservedOut->getScalarMemoryArea("E")->getPointer()[0],
        1e-8);

    // Run twice to make sure it does not overwrite anything
    conservedOutCUDA->makeZero();

    functional->operator ()(*conservedOutCUDA, *conservedInCUDA, 0.5,
        grid);

    conservedOutCUDA->copyTo(*conservedOut);

    ASSERT_NEAR(0.0, conservedOut->getScalarMemoryArea("E")->getPointer()[0],
        1e-8);

}

TEST(EntropyTest, CUDAResultIsOneTest) {
    alsfvm::functional::FunctionalFactory factory;
    alsfvm::functional::Functional::Parameters parameters({{"gamma", "1.4"}, {"something", "something"}});

    auto functional = factory.makeFunctional("cuda", "log_entropy", parameters);

    ivec3 size = {2, 2, 1};
    auto conservedIn = alsfvm::volume::makeConservedVolume("cpu", "euler2", size,
            2);
    auto conservedInCUDA = alsfvm::volume::makeConservedVolume("cuda", "euler2",
            size,
            2);

    grid::Grid grid({0, 0, 0}, {1, 1, 1}, size);
    volume::fill_volume<equation::euler::ConservedVariables<2> >(*conservedIn,
    grid, [](real x, real y, real z, equation::euler::ConservedVariables<2>& out) {
        out.rho = exp(1.0);
        out.m.x = sin(2 * M_PI * x * y);
        out.m.y = sin(2 * M_PI * x * y);
        out.E = exp(1.4 + 0.4 / exp(1)) / 0.4 + .5 * out.m.dot(out.m) / out.rho;
    });

    auto conservedOut = alsfvm::volume::makeConservedVolume("cpu", "euler2", {1, 1, 1},
            0);
    auto conservedOutCUDA = alsfvm::volume::makeConservedVolume("cuda", "euler2", {1, 1, 1},
            0);
    conservedIn->copyTo(*conservedInCUDA);

    conservedOutCUDA->makeZero();

    functional->operator ()(*conservedOutCUDA, *conservedInCUDA, -1.0,
        grid);


    conservedOutCUDA->copyTo(*conservedOut);

    ASSERT_NEAR(1.0, conservedOut->getScalarMemoryArea("E")->getPointer()[0], 1e-8);
    // Run twice to make sure it does not overwrite anything
    conservedOutCUDA->makeZero();

    functional->operator ()(*conservedOutCUDA, *conservedInCUDA, -1.0,
        grid);


    conservedOutCUDA->copyTo(*conservedOut);

    ASSERT_NEAR(1.0, conservedOut->getScalarMemoryArea("E")->getPointer()[0], 1e-8);

}
#endif
