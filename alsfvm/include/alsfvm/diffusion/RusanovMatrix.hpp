#pragma once
#include "alsfvm/types.hpp"

namespace alsfvm {
    namespace diffusion {

        //! Represents the matrix
        //! \f[\Lambda = \max(|\lambda_i|)\mathbf{I},\f]
        //! where \f$\{\lambda_i\}\f$ are the Eigenvalues of the flux jacobian of 
        //! the system, and \f$\mathbf{F}\f$ is the identity matrix.
        template<class Equation, int direction>
        class RusanovMatrix {
        public:

            __device__ __host__ RusanovMatrix(const Equation& equation, const typename Equation::ConservedVariables& conservedVariables)
                : equation(equation), conservedVariables(conservedVariables)
            {
                // empty
            }

            template<typename VectorType>
            __device__ __host__ VectorType operator*(const VectorType& in) {
                VectorType out;
                auto eigenValues = equation.template computeEigenValues<direction>(conservedVariables);


                // find max eigenvalue
                real maxEigenValue = 0.0;
                for (size_t i = 0; i < eigenValues.size(); ++i) {
                    maxEigenValue = fmax(fabs(eigenValues[i]), maxEigenValue);
                }

                // multiply with max eigenvalue.s
                for (size_t i = 0; i < eigenValues.size(); ++i) {
                    out[i] = maxEigenValue * in[i];
                }

                return out;
            }

        private:
            const Equation& equation;
            typename Equation::ConservedVariables conservedVariables;
        };
    }
}