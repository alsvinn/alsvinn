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

#include <array>
#include "alsfvm/types.hpp"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include "alsutils/error/Exception.hpp"
#include <vector>

//! Simple function for doing linear fit.
//! \note THIS IS ONLY FOR UNITTESTS!

namespace alsfvm {
template<class T>
inline std::array<T, 2> linearFit(const std::vector<T>& x,
    const std::vector<T>& y) {
    const size_t N = x.size();

    // We will solve Ab = c, where
    //     [x_1   1]
    // A = [...  ..]
    //     [x_N   1]
    //
    // and
    //    [y_1]
    // c= [...]
    //    [y_N]
    boost::numeric::ublas::matrix<T> A(N, 1 + 1);
    boost::numeric::ublas::matrix<T> c(N, 1);

    for (size_t i = 0; i < N; ++i) {
        c(i, 0) = y[i];
        A(i, 1) = 1;
        A(i, 0) = x[i];
    }

    boost::numeric::ublas::matrix<T> AT(trans(A));
    boost::numeric::ublas::matrix<T> ATA(prec_prod(AT, A));
    boost::numeric::ublas::matrix<T> ATc(prec_prod(AT, c));

    boost::numeric::ublas::permutation_matrix<int> permutationMatrix(ATA.size1());

    if (boost::numeric::ublas::lu_factorize(ATA, permutationMatrix)) {
        THROW("Polyfit matrix singular");
    }

    boost::numeric::ublas::lu_substitute(ATA, permutationMatrix, ATc);

    return std::array < T, 1 + 1 > ({{ ATc(0, 0), ATc(1, 0) }});
}
}
