#pragma once
#include "alsfvm/equation/euler/ConservedVariables.hpp"
#include "alsfvm/equation/euler/ExtraVariables.hpp"
#include "alsfvm/equation/euler/AllVariables.hpp"
#include <cmath>
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
				typedef euler::ConservedVariables ConservedVariables;
				typedef euler::ExtraVariables ExtraVariables;

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



				///
				/// Computes the extra variables \f$ p \f$ and \f$u\f$. 
				/// Here we compute
				/// \f[u =\frac{1}{\rho} m\f]
				/// and
				/// \f[p = (1-\gamma)(E-\frac{1}{2\rho}m^2)\f]
				///
                static ExtraVariables computeExtra(const ConservedVariables& u) {
                    ExtraVariables v;
                    real ie = u.E - 0.5*u.m.dot(u.m)/u.rho;
                    v.u = u.m / u.rho;
                    v.p = (GAMMA-1)*ie;
                    return v;
                }

				///
				/// Computes the wave speed in the given direction
				/// (absolute value of wave speed)
				///
				template<int direction>
				static real computeWaveSpeed(const ConservedVariables& u,
					const ExtraVariables& v) {
					static_assert(direction >= 0, "Direction can not be negative");
					static_assert(direction < 3, "We only support dimension up to and inclusive 3");

                    return std::abs(v.u[direction]) + std::sqrt(GAMMA * v.p / u.rho);
				}

				/// 
				/// Checks to see if the variables obeys the constraint.
				/// In this case it checks that
				/// \f[\rho > 0\f]
				/// and
				/// \f[p\geq 0\f]
				/// 
				/// \returns true if the inequalities are fulfilled, false otherwise
				///
				static bool obeysConstraints(const ConservedVariables& u,
					const ExtraVariables& v) 
				{

                    return std::isfinite(u.rho) && (!std::isnan(u.rho)) && (u.rho > 0) && (v.p > 0);
				}

                static AllVariables makeAllVariables(real rho, real mx, real my, real mz, real E) {
                    assert(!std::isnan(rho));
                    assert(!std::isnan(mx));
                    assert(!std::isnan(my));
                    assert(!std::isnan(mz));
                    assert(!std::isnan(E));
                    ConservedVariables conserved(rho, mx, my, mz, E);
                    return AllVariables(conserved, computeExtra(conserved));
                }
			};
		}
} // namespace alsfvm
} // namespace equation

