#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/numflux/tecno_utils.hpp"
#include "alsfvm/equation/equation_list.hpp"
namespace alsfvm { namespace numflux { namespace euler { 

    
    //! Implements the entropy conservative flux found in the tecno
    //! paper (see
    //! http://www.cscamm.umd.edu/people/faculty/tadmor/pub/TV+entropy/Fjordholm_Mishra_Tadmor_SINUM2012.pdf
    //! )
    class Tecno1 {
    public:
        ///
        /// \brief name is "tecno1"
        ///
        static const std::string name;


        //! Computes the entropy conservative flux.
        //!
        template<int direction>
        __device__ __host__ inline static real computeFlux(const equation::euler::Euler& eq, const equation::euler::AllVariables& left,
            const equation::euler::AllVariables& right, equation::euler::ConservedVariables& F)
        {
            real gamma = eq.getGamma();
            auto rightZ = eq.computeTecnoVariables(right).z;
            auto leftZ = eq.computeTecnoVariables(left).z;
            if (left == right) {
                for (int i = 0; i < 5; ++i) {
                    F[i] = 0;
                }
            }
            else if (direction == 0) {
                F[0] = bar(leftZ[1], rightZ[1])*ln(leftZ[4], rightZ[4]);
                F[1] = bar(leftZ[4], rightZ[4])/bar(leftZ[0], rightZ[0]) + bar(leftZ[1], rightZ[1]) / bar(leftZ[0], rightZ[0]) * F[0];
                F[2] = bar(leftZ[2], rightZ[2]) / bar(leftZ[1], rightZ[1]) * F[0];
                F[3] = 0; // assume 2D
                F[4] = 1.0 / (2 * bar(leftZ[0], rightZ[0])) * ((gamma + 1) / (gamma - 1)*(F[0]) / (ln(leftZ[0], rightZ[0])) + bar(leftZ[1], rightZ[1])*F[1] + bar(leftZ[2], rightZ[2])*F[2]);
            } else  if (direction == 1) {
                F[0] = bar(leftZ[2], rightZ[2])*ln(leftZ[4], rightZ[4]);
                F[1] = bar(leftZ[1], rightZ[1]) / bar(leftZ[0], rightZ[0]) * F[0];
                F[2] = bar(leftZ[4], rightZ[4]) / bar(leftZ[0], rightZ[0]) + bar(leftZ[2], rightZ[2]) / bar(leftZ[0], rightZ[0]) * F[0];
                F[3] = 0; // assume 2D
                F[4] = 1.0 / (2 * bar(leftZ[0], rightZ[0])) * ((gamma + 1) / (gamma - 1)*(F[0]) / (ln(leftZ[0], rightZ[0])) + bar(leftZ[1], rightZ[1])*F[1] + bar(leftZ[2], rightZ[2])*F[2]);
            }


            return fmax(eq.computeWaveSpeed<direction>(left, left), eq.computeWaveSpeed<direction>(right, right));
        }
    };
} // namespace euler
} // namespace numflux
} // namespace alsfvm
