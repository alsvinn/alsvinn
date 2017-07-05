#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/numflux/burgers/Godunov.hpp"
namespace alsfvm {
    namespace numflux {

        //!
        //! This is the fourth order accurate combination of numerical flux, as found in the tecno paper
        //! See eg. http://www.cscamm.umd.edu/people/faculty/tadmor/pub/TV+entropy/Fjordholm_Mishra_Tadmor_SINUM2012.pdf
        //! (Fjordholm et al, Arbitrarily high-order accurate entropy stable essentially nonoscillatory schemes for systems of conservation laws)
        //!
        //! Concretely, we set
        //! \f[F_{i+1/2}^4:= \frac{4}{3}F(u_i, u_{i+1})-\frac{1}{6}\left(F(u_{i-1}, u_{i+1})+F(u_{i},u_{i+2})\right)\f]
        //! 
        //! for some given flux \f$F\f$. In our case, \f$F\f$ is always one of the entropy perserving fluxes.s
        //!
        template<class Equation, class BaseFlux>
        class TecnoCombined4 {
        public:
            ///
            /// \brief name is "tecno4"
            ///
            static const std::string name;

            template<int direction>
            __device__ __host__ inline static real computeFlux(const Equation& eq,
                const typename Equation::AllVariables& uiMinus1,
                const typename Equation::AllVariables& ui,
                const typename Equation::AllVariables& uiPlus1,
                const typename Equation::AllVariables& uiPlus2,
                typename Equation::ConservedVariables& F) {

                real maxWaveSpeed = 0;
                // helper function to compute the numerical flux
                auto flux = [&](const typename Equation::AllVariables& left,
                    const typename Equation::AllVariables& right
                    ) {

                    typename Equation::ConservedVariables returnValue;
                    real waveSpeed = BaseFlux::template computeFlux<direction>(eq, left, right, returnValue);
                    maxWaveSpeed = fmax(maxWaveSpeed, waveSpeed);

                    return returnValue;

                };
                F = (4.0 / 3.0 * flux(ui, uiPlus1) -1.0 / 6.0 * (flux(uiMinus1, uiPlus1) + flux(ui, uiPlus2)));
                return maxWaveSpeed;
            }
            
            static constexpr bool hasStencil = true;

            static __host__ __device__ ivec4 stencil() {
                return{ -1, 0, 1, 2 };
            }
        };
    } // namespace numflux
} // namespace alsfvm
