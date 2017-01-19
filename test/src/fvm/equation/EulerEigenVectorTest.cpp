#include <gtest/gtest.h>
#include "alsfvm/equation/euler/Euler.hpp"
#include "utils/polyfit.hpp"
#include "alsfvm/diffusion/RoeMatrix.hpp"

using namespace alsfvm;
using namespace alsfvm::equation::euler;

template<int D>
struct Dimension {
    typedef typename alsutils::Types<D>::rvec vec;
    typedef typename alsutils::Types<D>::matrix matrix;
    typedef typename alsutils::Types<D+2>::matrix state_matrix;
    typedef typename alsutils::Types<D+2>::rvec state_vec;

    typedef typename Euler<D>::ConservedVariables ConservedVariables;
    typedef typename Euler<D>::PrimitiveVariables PrimitiveVariables;
    typedef Euler<D> EquationType;
    static constexpr int d = D;


    static vec make_vector(real x, real y, real z);
};

template<>
rvec3 Dimension<3>::make_vector(real x, real y, real z) {
    return rvec3(x,y,z);
}

template<>
rvec2 Dimension<2>::make_vector(real x, real y, real z) {
    return rvec2(x,y);
}

template<>
rvec1 Dimension<1>::make_vector(real x, real y, real z) {
    return rvec1(x);
}


template<class Dim>
struct EulerEigenVectorTest : public ::testing::Test {
    static constexpr int d = Dim::d;

    typedef typename Dim::vec vec;
    typedef typename Dim::matrix matrix;
    typedef typename Dim::state_matrix state_matrix;
    typedef typename Dim::state_vec state_vec;

    typedef typename Dim::ConservedVariables ConservedVariables;
    typedef typename Dim::PrimitiveVariables PrimitiveVariables;
    typedef typename Dim::EquationType EquationType;


    EulerEigenVectorTest()
        : equation(parameters),
        gamma(parameters.getGamma()),
        gammaHat(gamma-1),
        primitiveVariables(rho, Dim::make_vector( u, v, w ), p),
        conservedVariables(equation.computeConserved(primitiveVariables)),
        E(conservedVariables.E),
        H((E + p) / rho),
        a(sqrtf(gamma*p / rho))
    {
        makeA();

    }

    void testJacobian() {
        // Should converge with rate 1

        std::vector<real> resolutions;
        std::vector<real> errors;

        for (int k = 3; k < 25; ++k) {
            real error = 0;
            int N = 2 << k;
            real h = 1.0 / N;
            state_matrix approx;
            for (int c = 0; c < d+2; ++c) {
                ConservedVariables delta;
                delta[c] = h;
                auto uPlusDelta = conservedVariables + delta;

                auto diff = (equation.template computePointFlux<0>(uPlusDelta) - this->equation.template computePointFlux<0>(this->conservedVariables)) / h;

                for (int j = 0; j < d+2; ++j) {
                    error += powf(this->A(j, c) - diff[j], 2);

                    approx(j, c) = diff[j];
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

    void testEigenValues() {
        // We test that the EigenVectors obeys
        // A*v=lambda v
        // where A is the flux jacobian

        // This is the flux jacobian of the x direction flux.
        // see Toro's book (page 108)


        auto eigenVectors = equation.template computeEigenVectorMatrix<0>(conservedVariables);
        auto eigenValues = equation.template computeEigenValues<0>(conservedVariables);
        for (int i = 0; i < d+2; ++i) {
            state_vec eigenVector;
            for (int j = 0; j < d+2; ++j) {
                eigenVector[j] = eigenVectors(j, i);
            }

            state_vec eigenVectorMultipliedByA = A*eigenVector;
            // First we see if it is an eigenvector
            state_vec scaling;
            for (int j = 0; j < d+2; ++j) {
                if (eigenVectorMultipliedByA[j] != 0) {
                    scaling[j] = eigenVectorMultipliedByA[j] / eigenVector[j];
                }
            }
            for (int j = 0; j < d+2; ++j) {
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

    void testPositive() {

        auto d = this->d;
        auto eigenVectors = equation.template computeEigenVectorMatrix<0>(conservedVariables);
        alsfvm::diffusion::RoeMatrix<EquationType, 0> roeMatrix(equation, conservedVariables);
        auto eigenVectorsTransposed = eigenVectors.transposed();
        state_matrix roeMatrixTimesEigenVectors;

        for (int column = 0; column < d+2; ++column) {
            state_vec columnVector;
            for (int row = 0; row < d+2; ++row) {
                columnVector[row] = eigenVectorsTransposed(row, column);
            }

            auto multiplied = roeMatrix * columnVector;

            for (int row = 0; row < d+2; ++row) {
                roeMatrixTimesEigenVectors(row, column) = multiplied[row];
            }
        }

        auto finalMatrix = eigenVectors * roeMatrixTimesEigenVectors;

        // check symmetric:
        for (int i = 0; i < d+2; ++i) {
            for (int j = 0; j < i; ++j) {
                ASSERT_FLOAT_EQ(finalMatrix(i, j), finalMatrix(j, i));
            }
        }


    }


    void testProduct() {


        auto eigenVectors = equation.template computeEigenVectorMatrix<0>(conservedVariables);
        alsfvm::diffusion::RoeMatrix<alsfvm::equation::euler::Euler<d>, 0> roeMatrix(equation, conservedVariables);
        auto eigenVectorsTransposed = eigenVectors.transposed();
        state_matrix roeMatrixTimesEigenVectors;

        for (int column = 0; column < d+2; ++column) {
            state_vec columnVector;
            for (int row = 0; row < d+2; ++row) {
                columnVector[row] = eigenVectorsTransposed(row, column);
            }

            auto multiplied = roeMatrix * columnVector;

            for (int row = 0; row < d+2; ++row) {
                roeMatrixTimesEigenVectors(row, column) = multiplied[row];
            }
        }

        auto finalMatrix = eigenVectors * roeMatrixTimesEigenVectors;

        state_vec vector;

        for (int i = 0; i < d; ++i) {
            vector[i] = i;
        }

        // check that (A*D*A.T v = (A*D*A.T)*v)

        auto vectorMultiplied1 = equation.template computeEigenVectorMatrix<0>(conservedVariables) * (roeMatrix * ((equation.template computeEigenVectorMatrix<0>(conservedVariables).transposed())*vector));
        auto vectorMultiplied2 = finalMatrix * vector;

        for (int i = 0; i < d+2; ++i) {
            ASSERT_FLOAT_EQ(vectorMultiplied1[i], vectorMultiplied2[i]);
        }

    }

    EulerParameters parameters;
    EquationType equation;
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
    state_matrix A;

    void makeA();
};

template<>
void EulerEigenVectorTest<Dimension<3> >::makeA() {
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


template<>
void EulerEigenVectorTest<Dimension<2> >::makeA() {
    A(0, 0) = 0;
    A(0, 1) = 1;
    A(0, 2) = 0;
    A(0, 3) = 0;

    A(1, 0) = gammaHat*H - u*u - a*a;
    A(1, 1) = (3 - gamma)*u;
    A(1, 2) = -gammaHat*v;
    A(1, 3) = gammaHat;

    A(2, 0) = -u*v;
    A(2, 1) = v;
    A(2, 2) = u;
    A(2, 3) = 0;


    auto m1 = conservedVariables.m.x;
    auto m2 = conservedVariables.m.y;
    A(3, 0) = 0.5*(gamma-1)*m1*(m1*m1+m2*m2)/(rho*rho*rho)-(m1*(E+(gamma-1)*(E-0.5*(m1*m1+m2*m2)/(rho))))/(rho*rho); //0.5*u*((gamma - 3)*H - a*a);
    //A(4, 0) = 0.5*u*((gamma - 3)*H - a*a);
    A(3, 1) = H - gammaHat*u*u;
    A(3, 2) = -gammaHat*u*v;
    A(3, 3) = gamma*u;

}

template<>
void EulerEigenVectorTest<Dimension<1> >::makeA() {
    A(0, 0) = 0;
    A(0, 1) = 1;
    A(0, 2) = 0;

    A(1, 0) = gammaHat*H - u*u - a*a;
    A(1, 1) = (3 - gamma)*u;
    A(1, 2) = gammaHat;



    auto m1 = conservedVariables.m.x;
    A(2, 0) = 0.5*(gamma-1)*m1*(m1*m1)/(rho*rho*rho)-(m1*(E+(gamma-1)*(E-0.5*(m1*m1)/(rho))))/(rho*rho); //0.5*u*((gamma - 3)*H - a*a);
    A(2, 1) = H - gammaHat*u*u;
    A(2, 2) = gamma*u;

}

TYPED_TEST_CASE_P(EulerEigenVectorTest);

TYPED_TEST_P(EulerEigenVectorTest, JacobianTest) {
    this->testJacobian();
}

TYPED_TEST_P(EulerEigenVectorTest, EigenValuesEigenVectors) {
   
    this->testEigenValues();
    
}

TYPED_TEST_P(EulerEigenVectorTest, PositiveDefiniteTest) {
    this->testPositive();
}

TYPED_TEST_P(EulerEigenVectorTest, ProductTest) {

    this->testProduct();
}

REGISTER_TYPED_TEST_CASE_P(EulerEigenVectorTest, ProductTest, PositiveDefiniteTest,
                           EigenValuesEigenVectors, JacobianTest);

typedef ::testing::Types<Dimension<3>, Dimension<2>, Dimension<1> > MyTypes;
INSTANTIATE_TYPED_TEST_CASE_P(EulerEigenTests, EulerEigenVectorTest, MyTypes);
