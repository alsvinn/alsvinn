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
    functional->operator ()(*conservedOut, *extraOut, *conservedIn, *extraIn, 0.5,
        grid);

    for (size_t var = 0; var < conservedOut->getNumberOfVariables(); ++var) {
        ASSERT_DOUBLE_EQ(0.5, conservedOut->getScalarMemoryArea(var)->getPointer()[0]);
    }

    for (size_t var = 0; var < extraOut->getNumberOfVariables(); ++var) {
        ASSERT_DOUBLE_EQ(0.5, extraOut->getScalarMemoryArea(var)->getPointer()[0]);
    }


}
