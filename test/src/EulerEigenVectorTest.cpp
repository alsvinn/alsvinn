#include <gtest/gtest.h>
#include "alsfvm/equation/euler/Euler.hpp"
#include "utils/polyfit.hpp"
#include "alsfvm/diffusion/RoeMatrix.hpp"
using namespace alsfvm;
using namespace alsfvm::equation::euler;

struct EulerEigenVectorTest : public ::testing::Test {
    EulerEigenVectorTest()
        : equation(parameters),
        gamma(parameters.getGamma()),
        gammaHat(gamma-1),
        primitiveVariables(rho, rvec3{ u, v, w }, p),
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
        auto m1 = conservedVariables.m.x;
        auto m2 = conservedVariables.m.y;
        auto m3 = conservedVariables.m.z;
        A(4, 0) = 0.5*(gamma-1)*m1*(m1*m1+m2*m2+m3*m3)/(rho*rho*rho)-(m1*(E+(gamma-1)*(E-0.5*(m1*m1+m2*m2+m3*m3)/(rho))))/(rho*rho); //0.5*u*((gamma - 3)*H - a*a);
        //A(4, 0) = 0.5*u*((gamma - 3)*H - a*a);
        A(4, 1) = H - gammaHat*u*u;
        A(4, 2) = -gammaHat*u*v;
        A(4, 3) = -gammaHat*u*w;
        A(4, 4) = gamma*u;
    }

    EulerParameters parameters;
    Euler<3> equation;
    const real gamma;
    const real gammaHat;

    const double v = 1.4;
    const double u = 1.42;
    const double w = 1.32;

    const double rho = 3;
    const double p = 5;

    PrimitiveVariables<3> primitiveVariables;
    ConservedVariables<3> conservedVariables;
    real E;
    real H;
    real a;
    matrix5 A;
};

TEST_F(EulerEigenVectorTest, JacobianTest) {
    // Should converge with rate 1

    std::vector<real> resolutions;
    std::vector<real> errors;

    for (int k = 3; k < 25; ++k) {
        real error = 0;
        int N = 2 << k;
        real h = 1.0 / N;
        matrix5 approx;
        for (int d = 0; d < 5; ++d) {
            ConservedVariables<3> delta;
            delta[d] = h;
            auto uPlusDelta = conservedVariables + delta;

            auto diff = (equation.computePointFlux<0>(uPlusDelta) - equation.computePointFlux<0>(conservedVariables)) / h;

            for (int j = 0; j < 5; ++j) {
                error += powf(A(j, d) - diff[j], 2);

                approx(j, d) = diff[j];
            }
        }
        if (k == 5 || k == 10 || k == 24) {
          //  std::cout << "Approx = " << approx << std::endl;
           // std::cout << "Real   = " << A << std::endl;
        }
        //std::cout << "error = " << error << std::endl;
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
            EXPECT_NEAR(eigenValues[i] * eigenVector[j], eigenVectorMultipliedByA[j], 1e-6)
                << "Mismatch eigenvector " << i << ", component " << j << std::endl
                << "\teigenVector = " << eigenVector << std::endl
                << "\teigenValue  = " << eigenValues[i] << std::endl
                << "\tmultiplied  = " << eigenValues[i] * eigenVector << std::endl
                << "\tresult      = " << eigenVectorMultipliedByA << std::endl
                << "\tscalings    = " << scaling << std::endl;
        }
    }
}

TEST_F(EulerEigenVectorTest, PositiveDefiniteTest) {
    auto eigenVectors = equation.computeEigenVectorMatrix<0>(conservedVariables);
    alsfvm::diffusion::RoeMatrix<alsfvm::equation::euler::Euler<3>, 0> roeMatrix(equation, conservedVariables);
    auto eigenVectorsTransposed = eigenVectors.transposed();
    matrix5 roeMatrixTimesEigenVectors;

    for (int column = 0; column < 5; ++column) {
        rvec5 columnVector;
        for (int row = 0; row < 5; ++row) {
            columnVector[row] = eigenVectorsTransposed(row, column);
        }

        auto multiplied = roeMatrix * columnVector;

        for (int row = 0; row < 5; ++row) {
            roeMatrixTimesEigenVectors(row, column) = multiplied[row];
        }
    }

    auto finalMatrix = eigenVectors * roeMatrixTimesEigenVectors;

    // check symmetric:
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < i; ++j) {
            ASSERT_FLOAT_EQ(finalMatrix(i, j), finalMatrix(j, i));
        }
    }


}

TEST_F(EulerEigenVectorTest, ProductTest) {
    auto eigenVectors = equation.computeEigenVectorMatrix<0>(conservedVariables);
    alsfvm::diffusion::RoeMatrix<alsfvm::equation::euler::Euler<3>, 0> roeMatrix(equation, conservedVariables);
    auto eigenVectorsTransposed = eigenVectors.transposed();
    matrix5 roeMatrixTimesEigenVectors;

    for (int column = 0; column < 5; ++column) {
        rvec5 columnVector;
        for (int row = 0; row < 5; ++row) {
            columnVector[row] = eigenVectorsTransposed(row, column);
        }

        auto multiplied = roeMatrix * columnVector;

        for (int row = 0; row < 5; ++row) {
            roeMatrixTimesEigenVectors(row, column) = multiplied[row];
        }
    }

    auto finalMatrix = eigenVectors * roeMatrixTimesEigenVectors;

    rvec5 vector{ 1,2,3,4,5 };

    // check that (A*D*A.T v = (A*D*A.T)*v)

    auto vectorMultiplied1 = equation.template computeEigenVectorMatrix<0>(conservedVariables) * (roeMatrix * ((equation.template computeEigenVectorMatrix<0>(conservedVariables).transposed())*vector));
    auto vectorMultiplied2 = finalMatrix * vector;

    for (int i = 0; i < 5; ++i) {
        ASSERT_FLOAT_EQ(vectorMultiplied1[i], vectorMultiplied2[i]);
    }
}