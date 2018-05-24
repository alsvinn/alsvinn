/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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

    __device__ __host__ RusanovMatrix(const Equation& equation,
        const typename Equation::ConservedVariables& conservedVariables)
        : equation(equation), conservedVariables(conservedVariables) {
        // empty
    }

    template<typename VectorType>
    __device__ __host__ VectorType operator*(const VectorType& in) {
        VectorType out;
        auto eigenValues = equation.template computeEigenValues<direction>
            (conservedVariables);


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