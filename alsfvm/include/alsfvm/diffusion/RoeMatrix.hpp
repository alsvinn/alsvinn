#pragma once
#include "alsfvm/types.hpp"

namespace alsfvm {
    namespace diffusion {

        //! Represents the matrix
        //! \f[\Lambda = \mathrm{diag}(\lambda_1,\ldots, \lambda_N),\f]
        //! where \f$\{\lambda_i\}\f$ are the Eigenvalues of the flux jacobian of 
        //! the system.
        template<class Equation, int direction>
        class RoeMatrix {
        public:
            
            __device__ __host__ RoeMatrix (const Equation& equation, const typename Equation::ConservedVariables& conservedVariables) 
                : equation(equation), conservedVariables(conservedVariables)
            {
                // empty
            }

            template<typename VectorType> 
            __device__ __host__ VectorType operator*(const VectorType& in) {
                VectorType out;
                auto eigenValues = equation.template computeEigenValues<direction>(conservedVariables);

                for (size_t i = 0; i < eigenValues.size(); ++i) {
                    out[i] = fabs(eigenValues[i]) * in[i];
                }

                return out;
            }

        private:
            const Equation& equation;
            typename Equation::ConservedVariables conservedVariables;
        };
    }
}