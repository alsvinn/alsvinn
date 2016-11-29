#include <gtest/gtest.h>
#include "alsfvm/equation/euler/Euler.hpp"
#include "utils/polyfit.hpp"
using namespace alsfvm;
using namespace alsfvm::equation::euler;

struct EulerEigenVectorTest : public ::testing::Test {
    EulerEigenVectorTest()
        : equation(parameters),
        gamma(parameters.getGamma()),
        gammaHat(gamma-1),
        primitiveVariables(rho, u, v, w, p),
        conservedVariables(equation.computeConserved(primitiveVariables)),
        E(conservedVariables.E),
        H((E + p) / rho),
        a(sqrtf(gamma*p / rho))
    {
        A(0, 0) = 0;
        A(0, 1) = 1;
        A(0, 2) = 0;
        A(0, 3) = 0;
        A(0, 4) = 0;

        A(1, 0) = gammaHat*H - u*u - a*a;
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
    }

    EulerParameters parameters;
    Euler equation;
    const real gamma;
    const real gammaHat;

    const double v = 1.4;
    const double u = 1.42;
    const double w = 1.32;

    const double rho = 3;
    const double p = 5;

    PrimitiveVariables primitiveVariables;
    ConservedVariables conservedVariables;
    real E;
    real H;
    real a;
    matrix5 A;
};

TEST_F(EulerEigenVectorTest, JacobianTest) {
    // Should converge with rate 1

    std::vector<real> resolutions;
    std::vector<real> errors;

    for (int k = 3; k < 30; ++k) {
        real error = 0;
        int N = 2 << k;
        real h = 1.0 / N;
        matrix5 approx;
        for (int d = 0; d < 5; ++d) {
            ConservedVariables delta;
            delta[d] = h;
            auto uPlusDelta = conservedVariables + delta;

            auto diff = (equation.computePointFlux<0>(uPlusDelta) - equation.computePointFlux<0>(conservedVariables)) / h;

            for (int j = 0; j < 5; ++j) {
                error += powf(A(j, d) - diff[j], 2);

                approx(j, d) = diff[j];
            }
        }
        if (k == 5 || k == 10 || k == 29) {
            std::cout << "Approx = " << approx << std::endl;
            std::cout << "Real   = " << A << std::endl;
        }
        std::cout << "error = " << error << std::endl;
        resolutions.push_back(std::log(h));
        errors.push_back(std::log(sqrtf(error)));

    }

    ASSERT_GE(linearFit(resolutions, errors)[0], 0.9);
}

TEST_F(EulerEigenVectorTest, EigenValuesEigenVectors) {
   

    

  
   
    // We test that the EigenVectors obeys
    // A*v=lambda v
    // where A is the flux jacobian
   
    // This is the flux jacobian of the x direction flux.
    // see Toro's book (page 108)


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