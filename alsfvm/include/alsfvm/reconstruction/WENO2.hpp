#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/reconstruction/WENOCoefficients.hpp"

namespace alsfvm { namespace reconstruction { 

///
/// Simple WENO2 reconstruction. This is to be used with the ReconstructionCPU
/// or ReconstructionCUDA class!
///
    template<class Equation>
    class WENO2 {
    public:

        __device__ __host__ static void reconstruct(typename Equation::ConstViews& in,
                                             size_t x, size_t y, size_t z,
                                             typename Equation::Views& leftView,
                                             typename Equation::Views& rightView,
                                             bool xDir, bool yDir, bool zDir)
        {
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


            typename Equation::ConservedVariables inLeft = Equation::fetchConservedVariables(in, indexLeft);
            typename Equation::ConservedVariables inMiddle = Equation::fetchConservedVariables(in, indexOut);
            typename Equation::ConservedVariables inRight = Equation::fetchConservedVariables(in, indexRight);

            typename Equation::ConservedVariables left =  0.5*(w1 * inLeft +
                                                               (3 * w0 + w1) * inMiddle -
                                                               w0 * inRight);

            typename Equation::ConservedVariables right =  0.5*(w0 * inRight +
                                                                (3 * w1 + w0) * inMiddle -
                                                                w1 * inLeft);
            Equation::setViewAt(leftView, indexOut, left);
            Equation::setViewAt(rightView, indexOut, right);
        }

        __device__ __host__ static size_t getNumberOfGhostCells() {
            return 2;
        }
    };
} // namespace alsfvm
} // namespace reconstruction
