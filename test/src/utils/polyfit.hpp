#include <array>
#include "alsfvm/types.hpp"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include "alsutils/error/Exception.hpp"
#include <vector>

//! Simple function for doing linear fit. 
//! \note THIS IS ONLY FOR UNITTESTS!

namespace alsfvm {
    inline std::array<real, 2> linearFit(const std::vector<real>& x,
        const std::vector<real>& y)
    {
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
        boost::numeric::ublas::matrix<real> A(N, 1 + 1);
        boost::numeric::ublas::matrix<real> c(N, 1);
        for (size_t i = 0; i < N; ++i) {
            c(i, 0) = y[i];
            A(i, 1) = 1;
            A(i, 0) = x[i];
        }

        boost::numeric::ublas::matrix<real> AT(trans(A));
        boost::numeric::ublas::matrix<real> ATA(prec_prod(AT, A));
        boost::numeric::ublas::matrix<real> ATc(prec_prod(AT, c));

        boost::numeric::ublas::permutation_matrix<int> permutationMatrix(ATA.size1());

        if (boost::numeric::ublas::lu_factorize(ATA, permutationMatrix)) {
            THROW("Polyfit matrix singular");
        }

        boost::numeric::ublas::lu_substitute(ATA, permutationMatrix, ATc);

        return std::array<real, 1 + 1>({ ATc(0, 0), ATc(1, 0) });
    }
}
