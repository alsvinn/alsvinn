#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/reconstruction/WENOCoefficients.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include <algorithm>

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
        __device__ __host__ static void reconstruct(Equation eq, typename Equation::ConstViews& in,
                                             size_t x, size_t y, size_t z,
                                             typename Equation::Views& leftView,
                                             typename Equation::Views& rightView,
                                             bool xDir, bool yDir, bool zDir)
        {
            const real BOUND = 1.9;
            const size_t indexOut = leftView.index(x, y, z);
            const size_t indexRight = leftView.index(x + xDir, y + yDir, z + zDir);
            const size_t indexLeft = leftView.index(x - xDir, y - yDir, z - zDir);
            const real i0 = eq.getWeight(in, indexOut);
            const real b0 = square(eq.getWeight(in, indexRight) - i0);
            const real b1 = square(i0 - eq.getWeight(in, indexLeft));
            const real a0 = 1 / (3 * (ALSFVM_WENO_EPSILON + b0)*(ALSFVM_WENO_EPSILON + b0));
            const real a1 = 2 / (3 * (ALSFVM_WENO_EPSILON + b1)*(ALSFVM_WENO_EPSILON + b1));
            const real w0 = a0 / (a0 + a1);
            const real w1 = a1 / (a0 + a1);


            typename Equation::PrimitiveVariables inLeft =
                    eq.computePrimitiveVariables(eq.fetchConservedVariables(in, indexLeft));

            typename Equation::PrimitiveVariables inMiddle =
                    eq.computePrimitiveVariables(eq.fetchConservedVariables(in, indexOut));

            typename Equation::PrimitiveVariables inRight =
                    eq.computePrimitiveVariables(eq.fetchConservedVariables(in, indexRight));

            typename Equation::PrimitiveVariables dP = w1 * (inMiddle - inLeft) +
                w0 * (inRight - inMiddle);


            dP.rho = std::max(-BOUND*inMiddle.rho, std::min(BOUND*inMiddle.rho, dP.rho));
            dP.p = std::max(-BOUND*inMiddle.p, std::min(BOUND*inMiddle.p, dP.p));

            real LL = 0.125*inMiddle.rho*(dP.u.dot(dP.u)) -
                0.5*std::min(0.0, dP.rho * (inMiddle.u.dot(dP.u))) +
                0.5*dP.rho*dP.rho*dP.u.dot(dP.u)/inMiddle.rho;


            real R = inMiddle.p / (eq.getGamma()-1);
            real aijk = 0.5*std::sqrt(R/std::max(R,LL));


            typename Equation::ConservedVariables left = eq.computeConserved(inMiddle - aijk * dP);

            typename Equation::ConservedVariables right =  eq.computeConserved(inMiddle + aijk * dP);
            eq.setViewAt(leftView, indexOut, left);
            eq.setViewAt(rightView, indexOut, right);

        }

        __device__ __host__ static size_t getNumberOfGhostCells() {
            return 2;
        }
    };
} // namespace alsfvm
} // namespace reconstruction
