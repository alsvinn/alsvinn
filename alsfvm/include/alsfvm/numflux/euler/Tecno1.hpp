#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/numflux/tecno_utils.hpp"
#include "alsfvm/equation/equation_list.hpp"
#include "alsfvm/numflux/euler/HLL3.hpp"
namespace alsfvm {
    namespace numflux {
        namespace euler {


            //! Implements the entropy conservative flux found in the tecno
            //! paper (see
            //! http://www.cscamm.umd.edu/people/faculty/tadmor/pub/TV+entropy/Fjordholm_Mishra_Tadmor_SINUM2012.pdf
            //! )
            template<int nsd>
            class Tecno1 {
            public:
                ///
                /// \brief name is "tecno1"
                ///
                static const std::string name;

                typedef typename Types<nsd>::rvec rvec;
                typedef typename Types<nsd + 2>::rvec state_vector;

                //! Computes the entropy conservative flux.
                //!
                template<int direction>
                __device__ __host__ inline static real computeFlux(const equation::euler::Euler<nsd>& eq, const equation::euler::AllVariables<nsd>& left,
                    const equation::euler::AllVariables<nsd>& right, equation::euler::ConservedVariables<nsd>& F);
            };

            //! Computes the entropy conservative flux.
            //!
            template<>
            template<int direction>
            __device__ __host__ inline  real Tecno1<3>::computeFlux(const equation::euler::Euler<3>& eq, const equation::euler::AllVariables<3>& left,
                const equation::euler::AllVariables<3>& right, equation::euler::ConservedVariables<3>& F)
            {
                real gamma = eq.getGamma();
                auto rightZ = eq.computeTecnoVariables(right).z;
                auto leftZ = eq.computeTecnoVariables(left).z;
                if (left == right) {
                    eq.computePointFlux<direction>(left, F);
                }
                else if (direction == 0) {
                    F[0] = bar(leftZ[1], rightZ[1])*ln(leftZ[4], rightZ[4]);
                    F[1] = bar(leftZ[4], rightZ[4]) / bar(leftZ[0], rightZ[0]) + bar(leftZ[1], rightZ[1]) / bar(leftZ[0], rightZ[0]) * F[0];
                    F[2] = bar(leftZ[2], rightZ[2]) / bar(leftZ[0], rightZ[0])* F[0];
                    F[3] = 0; // assume 2D
                    F[4] = 1.0 / (2 * bar(leftZ[0], rightZ[0])) * ((gamma + 1) / (gamma - 1)*divLn(leftZ[0], rightZ[0], (F[0])) + bar(leftZ[1], rightZ[1])*F[1] + bar(leftZ[2], rightZ[2])*F[2]);
                }
                else  if (direction == 1) {
                    F[0] = bar(leftZ[2], rightZ[2])*ln(leftZ[4], rightZ[4]);
                    F[1] = bar(leftZ[1], rightZ[1]) / bar(leftZ[0], rightZ[0]) * F[0];
                    F[2] = bar(leftZ[4], rightZ[4]) / bar(leftZ[0], rightZ[0]) + bar(leftZ[2], rightZ[2]) / bar(leftZ[0], rightZ[0]) * F[0];
                    F[3] = 0; // assume 2D
                    F[4] = 1.0 / (2 * bar(leftZ[0], rightZ[0])) * ((gamma + 1) / (gamma - 1)*divLn(leftZ[0], rightZ[0], (F[0])) + bar(leftZ[1], rightZ[1])*F[1] + bar(leftZ[2], rightZ[2])*F[2]);
                }

                real leftSpeed = 0, rightSpeed = 0;
                real cs = 0;
                HLL3<3>::computeHLLSpeeds<direction>(eq, left, right, leftSpeed, rightSpeed, cs);
                return fmax(fabs(leftSpeed), fabs(rightSpeed));

                //return fmax(eq.template computeWaveSpeed<direction>(left, left),
                //    eq.template computeWaveSpeed<direction>(right, right));
            }

            //! Computes the entropy conservative flux.
            //!
            template<>
            template<int direction>
            __device__ __host__ inline  real Tecno1<2>::computeFlux(const equation::euler::Euler<2>& eq, const equation::euler::AllVariables<2>& left,
                const equation::euler::AllVariables<2>& right, equation::euler::ConservedVariables<2>& F)
            {
                real gamma = eq.getGamma();
                auto rightZ = eq.computeTecnoVariables(right).z;
                auto leftZ = eq.computeTecnoVariables(left).z;
                if (left == right) {
                    eq.computePointFlux<direction>(left, F);
                }
                else if (direction == 0) {
                    F[0] = bar(leftZ[1], rightZ[1])*ln(leftZ[3], rightZ[3]);
                    F[1] = bar(leftZ[3], rightZ[3]) / bar(leftZ[0], rightZ[0]) + bar(leftZ[1], rightZ[1]) / bar(leftZ[0], rightZ[0]) * F[0];
                    F[2] = bar(leftZ[2], rightZ[2]) / bar(leftZ[0], rightZ[0])* F[0];
                    F[3] = 1.0 / (2 * bar(leftZ[0], rightZ[0])) * ((gamma + 1) / (gamma - 1)*divLn(leftZ[0], rightZ[0], (F[0])) + bar(leftZ[1], rightZ[1])*F[1] + bar(leftZ[2], rightZ[2])*F[2]);
                }
                else  if (direction == 1) {
                    F[0] = bar(leftZ[2], rightZ[2])*ln(leftZ[4], rightZ[4]);
                    F[1] = bar(leftZ[1], rightZ[1]) / bar(leftZ[0], rightZ[0]) * F[0];
                    F[2] = bar(leftZ[3], rightZ[3]) / bar(leftZ[0], rightZ[0]) + bar(leftZ[2], rightZ[2]) / bar(leftZ[0], rightZ[0]) * F[0];
                    F[3] = 1.0 / (2 * bar(leftZ[0], rightZ[0])) * ((gamma + 1) / (gamma - 1)*divLn(leftZ[0], rightZ[0], (F[0])) + bar(leftZ[1], rightZ[1])*F[1] + bar(leftZ[2], rightZ[2])*F[2]);
                }

                real leftSpeed = 0, rightSpeed = 0;
                real cs = 0;
                HLL3<2>::computeHLLSpeeds<direction>(eq, left, right, leftSpeed, rightSpeed, cs);
                return fmax(fabs(leftSpeed), fabs(rightSpeed));

                //return fmax(eq.template computeWaveSpeed<direction>(left, left),
                //    eq.template computeWaveSpeed<direction>(right, right));
            }


            template<>
            template<int direction>
            __device__ __host__ inline  real Tecno1<1>::computeFlux(const equation::euler::Euler<1>& eq, const equation::euler::AllVariables<1>& left,
                const equation::euler::AllVariables<1>& right, equation::euler::ConservedVariables<1>& F)
            {
                real gamma = eq.getGamma();
                auto rightZ = eq.computeTecnoVariables(right).z;
                auto leftZ = eq.computeTecnoVariables(left).z;
                if (left == right) {
                    eq.computePointFlux<direction>(left, F);
                }
                else if (direction == 0) {
                    F[0] = bar(leftZ[1], rightZ[1])*ln(leftZ[2], rightZ[2]);
                    F[1] = bar(leftZ[2], rightZ[2]) / bar(leftZ[0], rightZ[0]) + bar(leftZ[1], rightZ[1]) / bar(leftZ[0], rightZ[0]) * F[0];
                    F[2] = 1.0 / (2 * bar(leftZ[0], rightZ[0])) * ((gamma + 1) / (gamma - 1)*divLn(leftZ[0], rightZ[0], (F[0])) + bar(leftZ[1], rightZ[1])*F[1]);
                }

                real leftSpeed = 0, rightSpeed = 0;
                real cs = 0;
                HLL3<1>::computeHLLSpeeds<direction>(eq, left, right, leftSpeed, rightSpeed, cs);
                return fmax(fabs(leftSpeed), fabs(rightSpeed));

                //return fmax(eq.template computeWaveSpeed<direction>(left, left),
                //    eq.template computeWaveSpeed<direction>(right, right));
            }
        } // namespace euler
    } // namespace numflux
} // namespace alsfvm
