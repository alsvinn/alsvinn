#pragma once
#include "alsfvm/equation/euler/ConservedVariables.hpp"
#include "alsfvm/equation/euler/ExtraVariables.hpp"
#include "alsfvm/equation/euler/AllVariables.hpp"

///
/// Gamma constant
/// \note This will be moved into a paramter struct soon! 
///
#define GAMMA (5.0/3.0)


namespace alsfvm {
	namespace equation {
		namespace euler {

            class Euler {
			public:
				///
				/// Computes the point flux. 
				///
				/// Here we view the Euler equation as the following hyperbolic system				
				/// \f[\vec{U}_t+F(\vec{U})_x+G(\vec{U})_y+H(\vec{U})_z = 0\f]
				/// where 
				/// \f[\vec{U}=\left(\begin{array}{l}\rho\\ \vec{m}\\E\end{array}\right)=\left(\begin{array}{c}\rho\\ \rho u\\ \rho v\\ \rho w \\E\end{array}\right)\f]
				/// and
				/// \f[F(\vec{U})=\left(\begin{array}{c}\rho u\\ \rho u^2+p\\ \rho u v\\ \rho u w\\ u(E+p)\end{array}\right)\qquad
				///	   G(\vec{U})=\left(\begin{array}{c}\rho v\\ \rho uv\\ \rho v^2+p\\ \rho v w\\ v(E+p)\end{array}\right)\qquad
				///    H(\vec{U})=\left(\begin{array}{c}\rho w\\ \rho uw\\ \rho w v\\ \rho w^2+p\\ w(E+p)\end{array}\right)
				/// \f]
				/// \returns \f[\left\{\begin{array}{lr}F(\vec{U})&\mathrm{if}\;\mathrm{direction}=0\\
				///										G(\vec{U})&\mathrm{if}\;\mathrm{direction}=1\\
				///                                     H(\vec{U})&\mathrm{if}\;\mathrm{direction}=2
				///           \end{array}\right.\f]
				///
				/// \param[in] u the variables to use
				/// \param[out] F the resulting flux
				///

				template<size_t direction>
				static void computePointFlux(const AllVariables& u, ConservedVariables& F) {
					static_assert(direction < 3, "We only support up to three dimensions");

					F.rho = u.m[direction];
					F.m = u.u[direction] * u.m;
					F.m[direction] += u.p;
					F.E = (u.E + u.p) * u.u[direction];
				}

                typedef euler::ConservedVariables ConservedVariables;
                typedef euler::ExtraVariables ExtraVariables;

                static ExtraVariables computeExtra(const ConservedVariables& u) {
                    ExtraVariables v;
                    real ie = u.E - 0.5*u.m.dot(u.m)/u.rho;
                    v.u = u.m / Uijk.rho;
                    v.p = (GAMMA-1)*ie;
                    return v;
                }
			};
		}
} // namespace alsfvm
} // namespace equation

