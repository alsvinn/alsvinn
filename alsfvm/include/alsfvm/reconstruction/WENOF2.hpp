#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/reconstruction/WENOCoefficients.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
namespace alsfvm { namespace reconstruction { 
///
/// Simple WENO2 reconstruction. This is to be used with the ReconstructionCPU
/// or ReconstructionCUDA class!
///
/// This WENO2 implementation enforces the equation based constraints by computing
/// the reconstruction in the primitive variables and enforcing the natural bounds there.
///
    template<class Equation>
    class WENOF2 {
    public:
        __device__ __host__ static void reconstruct(typename Equation::ConstViews& in,
                                             size_t x, size_t y, size_t z,
                                             typename Equation::Views& leftView,
                                             typename Equation::Views& rightView,
                                             bool xDir, bool yDir, bool zDir)
        {
            const real BOUND = 1.9;
            const size_t indexOut = leftView.index(x, y, z);
            const size_t indexRight = leftView.index(x + xDir, y + yDir, z + zDir);
            const size_t indexLeft = leftView.index(x - xDir, y - yDir, z - zDir);
            const real i0 = Equation::getWeight(in, indexOut);
            const real b0 = square(Equation::getWeight(in, indexRight) - i0);
            const real b1 = square(i0 - Equation::getWeight(in, indexLeft));
            const real a0 = 1 / (3 * (ALSFVM_WENO_EPSILON + b0)*(ALSFVM_WENO_EPSILON + b0));
            const real a1 = 2 / (3 * (ALSFVM_WENO_EPSILON + b1)*(ALSFVM_WENO_EPSILON + b1));
            const real w0 = a0 / (a0 + a1);
            const real w1 = a1 / (a0 + a1);


            typename Equation::PrimitiveVariables inLeft =
                    Equation::computePrimitiveVariables(Equation::fetchConservedVariables(in, indexLeft));

            typename Equation::PrimitiveVariables inMiddle =
                    Equation::computePrimitiveVariables(Equation::fetchConservedVariables(in, indexOut));

            typename Equation::PrimitiveVariables inRight =
                    Equation::computePrimitiveVariables(Equation::fetchConservedVariables(in, indexRight));

            typename Equation::PrimitiveVariables dP = w1 * (inMiddle - inLeft) +
                w0 * (inRight - inMiddle);


            dP.rho = std::max(-BOUND*inMiddle.rho, std::min(BOUND*inMiddle.rho, dP.rho));
            dP.p = std::max(-BOUND*inMiddle.p, std::min(BOUND*inMiddle.p, dP.p));

            real LL = 0.125*inMiddle.rho*(dP.u.dot(dP.u)) -
                0.5*std::min(0.0, dP.rho * (inMiddle.u.dot(dP.u))) +
                0.5*dP.rho*dP.rho*dP.u.dot(dP.u)/inMiddle.rho;


            real R = inMiddle.p / (GAMMA-1);
            real aijk = 0.5*std::sqrt(R/std::max(R,LL));


            typename Equation::ConservedVariables left = Equation::computeConserved(inMiddle - aijk * dP);

            typename Equation::ConservedVariables right =  Equation::computeConserved(inMiddle + aijk * dP);
            Equation::setViewAt(leftView, indexOut, left);
            Equation::setViewAt(rightView, indexOut, right);
        }

        __device__ __host__ static size_t getNumberOfGhostCells() {
            return 2;
        }
    };
} // namespace alsfvm
} // namespace reconstruction
