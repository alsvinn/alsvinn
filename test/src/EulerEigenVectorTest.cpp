#include <gtest/gtest.h>
#include "alsfvm/equation/euler/Euler.hpp"

using namespace alsfvm;
using namespace alsfvm::equation::euler;

TEST(EulerEigenVectorTest, EigenValuesEigenVectors) {
    EulerParameters parameters;
    Euler equation(parameters);
    real gamma = parameters.getGamma();
    real gammaHat = gamma-1;

    double v = 1.4;
    double u = 1.42;
    double w = 1.32;

    double rho = 3;
    double p = 5;

    PrimitiveVariables primitiveVariables(rho, u, v, w, p);

    auto conservedVariables = equation.computeConserved(primitiveVariables);

    real E = conservedVariables.E;

    real H = (E + p) / rho;
    real a = sqrtf(gamma*p / rho);
    // We test that the EigenVectors obeys
    // A*v=lambda v
    // where A is the flux jacobian
    matrix5 A;
    // This is the flux jacobian of the x direction flux.
    // see Toro's book (page 108)
    A(0, 0) = 0;
    A(0, 1) = 1;
    A(0, 2) = 0;
    A(0, 3) = 0;
    A(0, 4) = 0;

    A(1, 0) = gammaHat*H-u*u-a*a;
    A(1, 1) = (3 - gamma)*u;
    A(1, 2) = -gammaHat*v;
    A(1, 3) = -gammaHat*w;
    A(1, 4) = gammaHat;

    A(2, 0) = -u*v;
    A(2, 1) = v;
    A(2, 2) = u;
    A(2, 3) = 0;
    A(2, 4) = 0;

    A(3, 0) = -u*w;
    A(3, 1) = w;
    A(3, 2) = 0;
    A(3, 3) = u;
    A(3, 4) = 0;

    A(4, 0) = 0.5*u*((gamma - 3)*H - a*a);
    A(4, 1) = H - gammaHat*u*u;
    A(4, 2) = -gammaHat*u*v;
    A(4, 3) = -gammaHat*u*w;
    A(4, 4) = gamma*u;

    auto eigenVectors = equation.computeEigenVectorMatrix<0>(conservedVariables);
    auto eigenValues = equation.computeEigenValues<0>(conservedVariables);
    for (int i = 0; i < 5; ++i) {
        rvec5 eigenVector;
        for (int j = 0; j < 5; ++j) {
            eigenVector[j] = eigenVectors(j, i);
        }

        rvec5 eigenVectorMultipliedByA = A*eigenVector;
        // First we see if it is an eigenvector
        rvec5 scaling;
        for (int j = 0; j < 5; ++j) {
            if (eigenVectorMultipliedByA[j] != 0) {
                scaling[j] = eigenVectorMultipliedByA[j] / eigenVector[j];
            }
        }
        for (int j = 0; j < 5; ++j) {
            EXPECT_NEAR(eigenValues[i] * eigenVector[j], eigenVectorMultipliedByA[j], 1e-5)
                << "Mismatch eigenvector " << i << ", component " << j << std::endl
                << "\teigenVector = " << eigenVector << std::endl
                << "\teigenValue  = " << eigenValues[i] << std::endl
                << "\tmultiplied  = " << eigenValues[i] * eigenVector << std::endl
                << "\tresult      = " << eigenVectorMultipliedByA << std::endl
                << "\tscalings    = " << scaling << std::endl;
        }
    }
}