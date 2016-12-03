#pragma once
#include "alsfvm/equation/euler/ConservedVariables.hpp"
#include "alsfvm/equation/euler/ExtraVariables.hpp"
#include "alsfvm/equation/euler/AllVariables.hpp"
#include "alsfvm/equation/euler/PrimitiveVariables.hpp"
#include "alsfvm/equation/euler/TecnoVariables.hpp"

#include "alsfvm/equation/euler/Views.hpp"
#include "alsfvm/volume/Volume.hpp"
#include "alsfvm/equation/euler/ViewsExtra.hpp"

#include "alsfvm/equation/euler/EulerParameters.hpp"
#include <iostream>
namespace alsfvm {
	namespace equation {
        namespace euler {

            //! Implements the euler equations
            //! @param nsd the number of space dimensions
            template<int nsd>
            class Euler {
            public:

                Euler(const EulerParameters& parameters)
                    : gamma(parameters.getGamma())
                {
                }

                typedef typename Types<nsd>::rvec vec;
                typedef typename Types<nsd + 2>::rvec state_vector;
                typedef typename Types<nsd + 2>::matrix state_matrix;
                typedef euler::EulerParameters Parameters;
                typedef typename euler::ConservedVariables<nsd> ConservedVariables;
                typedef typename euler::ExtraVariables<nsd> ExtraVariables;
                typedef typename euler::PrimitiveVariables<nsd> PrimitiveVariables;
                typedef typename euler::AllVariables<nsd> AllVariables;
                typedef typename euler::TecnoVariables<nsd> TecnoVariables;

                ///
                /// Defaults to "euler<nsd>".
                ///
                static const std::string name;

                //!
                //! List of all conserved variables used by Euler (rho, mx, my, mz, E)
                //!
                static const std::vector<std::string> conservedVariables;

                //!
                //! List of all primtive variables used by Euler (rho, vx, vy, vz, p)
                //!
                static const std::vector<std::string> primitiveVariables;

                //!
                //! List of all extra variables used by Euler (p, vx, vy, vz)
                //!
                static const std::vector<std::string> extraVariables;


                ///
                /// Gives the number of conserved variables used (2+nsd)
                ///
                static const size_t  numberOfConservedVariables = 2 + nsd;

                __device__ __host__ static size_t getNumberOfConservedVariables() {
                    return 2 + nsd;
                }

                typedef equation::euler::Views<volume::Volume, memory::View<real>, nsd > Views;
                typedef equation::euler::Views<const volume::Volume, const memory::View<const real>, nsd > ConstViews;

                typedef equation::euler::ViewsExtra<volume::Volume, memory::View<real>, nsd > ViewsExtra;
                typedef equation::euler::ViewsExtra<const volume::Volume, const memory::View<const real>, nsd > ConstViewsExtra;


                ///
                /// Fetches and computes the all variables from memory
                ///
                __device__ __host__ AllVariables fetchAllVariables(ConstViews& views, size_t index) const {
                    return makeAllVariables(views.rho.at(index),
                        views.m(index).template convert<real>(),
                        views.E.at(index));
                }

                template<class T, class S>
                __device__ __host__ static ConservedVariables fetchConservedVariables(euler::Views<T, S, nsd>& views, size_t index) {
                    return ConservedVariables(views.rho.at(index),
                        views.m(index),
                        views.E.at(index));
                }

                __device__ __host__ ExtraVariables fetchExtraVariables(ConstViewsExtra& views, size_t index) const {
                    return ExtraVariables(views.p.at(index),
                        views.u(index).template convert<real>()
                        );
                }

                ///
                /// Writes the ConservedVariable struct back to memory
                ///
                __device__ __host__ static void setViewAt(Views& output, size_t index, const ConservedVariables& input) {
                    output.rho.at(index) = input.rho;
                    output.m(index) = input.m;
                    output.E.at(index) = input.E;

                }

                ///
                /// Writes the ExtraVariable struct back to memory
                ///
                __device__ __host__ void setExtraViewAt(ViewsExtra& output, size_t index, const ExtraVariables& input) const {
                    output.p.at(index) = input.p;
                    output.u(index) = input.u;
                }

                ///
                /// Adds the conserved variables to the view at the given index
                /// 
                /// Basically sets output[index] += input
                ///
                __device__ __host__ void addToViewAt(Views& output, size_t index, const ConservedVariables& input) const {
                    output.rho.at(index) += input.rho;
                    output.m(index) += input.m;
                    output.E.at(index) += input.E;

                }

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
                __device__ __host__  void computePointFlux(const AllVariables& u, ConservedVariables& F) const {
                    static_assert(direction < 3, "We only support up to three dimensions");

                    F.rho = u.m[direction];
                    F.m = u.u[direction] * u.m;
                    F.m[direction] += u.p;
                    F.E = (u.E + u.p) * u.u[direction];
                }

                //! Utility function that calls 
                //! \code{.cpp}
                //! computePointFlux<direction>(const AllVariables& u, ConservedVariables& F);
                //! \endcode
                template<size_t direction>
                __device__ __host__ ConservedVariables computePointFlux(const ConservedVariables& u) const {
                    auto all = makeAllVariables(u.rho, u.m, u.E);
                    ConservedVariables F;
                    computePointFlux<direction>(all, F);
                    return F;
                }


                ///
                /// Computes the extra variables \f$ p \f$ and \f$u\f$. 
                /// Here we compute
                /// \f[u =\frac{1}{\rho} m\f]
                /// and
                /// \f[p = (1-\gamma)(E-\frac{1}{2\rho}m^2)\f]
                ///
                __device__ __host__  ExtraVariables computeExtra(const ConservedVariables& u) const {
                    ExtraVariables v;
                    real ie = u.E - 0.5*u.m.dot(u.m) / u.rho;
                    v.u = u.m / u.rho;
                    v.p = (gamma - 1)*ie;
                    return v;
                }

                ///
                /// \brief computes the extra variables from the primitive ones
                ///
                /// \param primitiveVariables the primtive variables
                /// \return the computed all variables
                /// \note This implementation is not made for speed! Should only be
                /// used sparsely (eg. for initialization).
                ///
                __device__ __host__ ExtraVariables computeExtra(const PrimitiveVariables& primitiveVariables) const {
                    return ExtraVariables(primitiveVariables.p,
                        primitiveVariables.u
                        );
                }

                ///
                /// \brief computes the extra variables from the primitive ones
                ///
                /// \param primitiveVariables the primtive variables
                /// \return the computed all variables
                /// \note This implementation is not made for speed! Should only be
                /// used sparsely (eg. for initialization).
                ///
                __device__ __host__ ConservedVariables computeConserved(const PrimitiveVariables& primitiveVariables) const {
                    const vec m = primitiveVariables.rho * primitiveVariables.u;

                    const real E =
                        primitiveVariables.p / (gamma - 1)
                        + 0.5*primitiveVariables.rho*primitiveVariables.u.dot(primitiveVariables.u);

                    return ConservedVariables(primitiveVariables.rho, m, E);
                }

                ///
                /// Computes the wave speed in the given direction
                /// (absolute value of wave speed)
                ///
                template<int direction>
                __device__ __host__ real computeWaveSpeed(const ConservedVariables& u,
                    const ExtraVariables& v) const {
                    static_assert(direction >= 0, "Direction can not be negative");
                    static_assert(direction < 3, "We only support dimension up to and inclusive 3");

                    return fabs(v.u[direction]) + sqrt(gamma * v.p / u.rho);
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
                __device__ __host__ bool obeysConstraints(const ConservedVariables& u,
                    const ExtraVariables& v) const
                {
#ifdef __CUDA_ARCH__

                    return u.rho < INFINITY && (u.rho == u.rho) && (u.rho > 0) && (v.p > 0);
#else
                    return std::isfinite(u.rho) && (u.rho == u.rho) && (u.rho > 0) && (v.p > 0);
#endif
                }

                __device__ __host__ AllVariables makeAllVariables(real rho, vec m, real E) const {

                    ConservedVariables conserved(rho, m, E);
                    return AllVariables(conserved, computeExtra(conserved));
                }

                __device__ __host__ real getWeight(const ConstViews& in, size_t index) const {
                    return in.rho.at(index);
                }

                __device__ __host__ PrimitiveVariables computePrimitiveVariables(const ConservedVariables& conserved) const {
                    vec u = conserved.m / conserved.rho;
                    real ie = conserved.E - 0.5*conserved.m.dot(conserved.m) / conserved.rho;

                    real p = (gamma - 1)*ie;
                    return PrimitiveVariables(conserved.rho, u, p);
                }

                __device__ __host__ real getGamma() const {
                    return gamma;
                }

                //! see definition of \f$\vec{z}\f$ in
                //!
                //! http://www.cscamm.umd.edu/people/faculty/tadmor/pub/TV+entropy/Fjordholm_Mishra_Tadmor_SINUM2012.pdf
                //!
                //! eq (6.11). That is, we set
                //!
                //! \f[\vec{z} = \left(\begin{array}{l} \sqrt{\frac{\rho}{p}}\\ \sqrt{\frac{\rho}{p}}u\\ \sqrt{\frac{\rho}{p}}v\\ \sqrt{\rho p}\end{array}\right).\f]
                //!
                __device__ __host__ TecnoVariables computeTecnoVariables(const ConservedVariables& conserved) const {
                    PrimitiveVariables primitiveVariables = computePrimitiveVariables(conserved);


                    return TecnoVariables(sqrt(primitiveVariables.rho / primitiveVariables.p),
                        sqrt(primitiveVariables.rho / primitiveVariables.p)*primitiveVariables.u,
                        sqrt(primitiveVariables.rho*primitiveVariables.p));
                }


                /// 
                /// Computes the entropy variable \f$v(u)\f$ given by
                /// \f[v(u) = \partial_u E(u)\f]
                /// corresponding to the entropy
                /// \f[E(u)=\frac{-\rho s}{\gamma-1}\f]
                /// Here \f$s\f$ is the thermodynamic entropy, given as 
                /// \f[s=\log(p)-\gamma \log (\rho)\f]
                ///
                /// The corresponding entropy variables are given as
                /// \f[v(\vec{u})=\left(\begin{array}{c}\frac{\gamma-s}{\gamma-1}-\frac{\rho(u_x^2+u_y^2+u_z^2)}{2p}\\
                ///                                      \rho u/p\\
                ///                                      \rho v/p\\
                ///                                      \rho w/p\\
                ///                                      -\rho/p\end{array}\right)\f]
                ///
                __device__ __host__ state_vector computeEntropyVariables(const ConservedVariables& conserved) const {
                    auto primitive = computePrimitiveVariables(conserved);
                    const real s = log(primitive.p) - gamma*log(conserved.rho);

                    return state_vector((gamma - s) / (gamma - 1) - (conserved.rho*(primitive.u.dot(primitive.u))) / (2 * primitive.p),
                        conserved.rho / primitive.p * primitive.u,
                        -conserved.rho / primitive.p);

                }

                ///
                /// Computes the entropy potential \f$\psi(u)\f$ given by
                /// \f[\psi(u) = v(u)f(u) - Q(u)\f]
                /// where \f$Q(u)\f$ is defined through 
                /// \f[Q'(u) = f'(u)E'(u)\f]
                /// Here
                /// \f[\psi(u) = \left(\begin{array}{c}\rho u\\ \rho v\\ \rho w\end{array}\right)=\vec{m}\f]
                ///
                __device__ __host__ vec computeEntropyPotential(const ConservedVariables& conserved) const {
                    return conserved.m;
                }


                template<int direction>
                __device__ __host__ state_vector computeEntropyVariablesMultipliedByEigenVectorMatrix(const ConservedVariables& conserved) const {

                    return computeEigenVectorMatrix<direction>(conserved).transposed()*computeEntropyVariables(conserved);
                }


                //! Computes the Eigen vector matrix. See 3.2.2 for full description in http://www.springer.com/de/book/9783540252023
                template<int direction>
                __device__ __host__ state_matrix computeEigenVectorMatrix(const ConservedVariables& conserved) const;

                //! Compute eigen values of the jacobian of \$F\f$, \f$G\f$ or \f$H\f$.
                //! See 3.2.2 for full description in http://www.springer.com/de/book/9783540252023
                template<int direction>
                __device__ __host__ state_vector computeEigenValues(const ConservedVariables& conserved) const;
            private:
                const real gamma;
            };


            template<>
            template<int direction>
            __device__ __host__ matrix5 Euler<3>::computeEigenVectorMatrix(const ConservedVariables& conserved) const {


                if (direction == 0) {
                    matrix5 matrixWithEigenVectors;

                    auto primitive = computePrimitiveVariables(conserved);
                    const real a = sqrt(gamma*primitive.p / conserved.rho);
                    const real H = (conserved.E + primitive.p) / conserved.rho;
                    matrixWithEigenVectors(0, 0) = 1;
                    matrixWithEigenVectors(0, 1) = 1;
                    matrixWithEigenVectors(0, 2) = 0;
                    matrixWithEigenVectors(0, 3) = 0;
                    matrixWithEigenVectors(0, 4) = 1;

                    matrixWithEigenVectors(1, 0) = primitive.u.x - a;
                    matrixWithEigenVectors(1, 1) = primitive.u.x;
                    matrixWithEigenVectors(1, 2) = 0;
                    matrixWithEigenVectors(1, 3) = 0;
                    matrixWithEigenVectors(1, 4) = primitive.u.x + a;

                    matrixWithEigenVectors(2, 0) = primitive.u.y;
                    matrixWithEigenVectors(2, 1) = primitive.u.y;
                    matrixWithEigenVectors(2, 2) = 1;
                    matrixWithEigenVectors(2, 3) = 0;
                    matrixWithEigenVectors(2, 4) = primitive.u.y;

                    matrixWithEigenVectors(3, 0) = primitive.u.z;
                    matrixWithEigenVectors(3, 1) = primitive.u.z;
                    matrixWithEigenVectors(3, 2) = 0;
                    matrixWithEigenVectors(3, 3) = 1;
                    matrixWithEigenVectors(3, 4) = primitive.u.z;

                    matrixWithEigenVectors(4, 0) = H - primitive.u.x*a;
                    matrixWithEigenVectors(4, 1) = 0.5*primitive.u.dot(primitive.u);
                    matrixWithEigenVectors(4, 2) = primitive.u.y;
                    matrixWithEigenVectors(4, 3) = primitive.u.z;
                    matrixWithEigenVectors(4, 4) = H + primitive.u.x*a;

                    return matrixWithEigenVectors.normalized();
                }
                else if (direction == 1) {
                    // We use the rotation trick, see 3.2.2 and Proposition 3.19 in Toro's book
                    // http://www.springer.com/de/book/9783540252023
                    auto conservedRotated = ConservedVariables(conserved.rho, rvec3{conserved.m.y, -conserved.m.x, conserved.m.z}, conserved.E);
                    return computeEigenVectorMatrix<0>(conservedRotated);
                }
                else if (direction == 2) {
                    // We use the rotation trick, see 3.2.2 and Proposition 3.19 in Toro's book
                    // http://www.springer.com/de/book/9783540252023
                    auto conservedRotated = ConservedVariables(conserved.rho, rvec3{conserved.m.z, conserved.m.y, -conserved.m.x}, conserved.E);
                    return computeEigenVectorMatrix<0>(conservedRotated);
                }

                assert(false);
            }

            template<>
            template<int direction>
            __device__ __host__ rvec5 Euler<3>::computeEigenValues(const ConservedVariables& conserved) const {
                if (direction == 0) {
                    auto primitive = computePrimitiveVariables(conserved);
                    const real a = sqrt(gamma*primitive.p / conserved.rho);
                    return rvec5(primitive.u.x - a, primitive.u.x, primitive.u.x, primitive.u.x, primitive.u.x + a);
                }
                else if (direction == 1) {
                    // We use the rotation trick, see 3.2.2 and Proposition 3.19 in Toro's book
                    // http://www.springer.com/de/book/9783540252023
                    auto conservedRotated = ConservedVariables(conserved.rho, rvec3{conserved.m.y, -conserved.m.x, conserved.m.z}, conserved.E);
                    return computeEigenValues<0>(conservedRotated);
                }
                else if (direction == 2) {
                    // We use the rotation trick, see 3.2.2 and Proposition 3.19 in Toro's book
                    // http://www.springer.com/de/book/9783540252023
                    auto conservedRotated = ConservedVariables(conserved.rho, rvec3{conserved.m.z, conserved.m.y, -conserved.m.x}, conserved.E);
                    return computeEigenValues<0>(conservedRotated);
                }
                assert(false);
            }



            template<>
            template<int direction>
            __device__ __host__ matrix4 Euler<2>::computeEigenVectorMatrix(const ConservedVariables& conserved) const {


                if (direction == 0) {
                    matrix4 matrixWithEigenVectors;

                    auto primitive = computePrimitiveVariables(conserved);
                    const real a = sqrt(gamma*primitive.p / conserved.rho);
                    const real H = (conserved.E + primitive.p) / conserved.rho;
                    matrixWithEigenVectors(0, 0) = 1;
                    matrixWithEigenVectors(0, 1) = 1;
                    matrixWithEigenVectors(0, 2) = 0;
                    matrixWithEigenVectors(0, 3) = 1;

                    matrixWithEigenVectors(1, 0) = primitive.u.x - a;
                    matrixWithEigenVectors(1, 1) = primitive.u.x;
                    matrixWithEigenVectors(1, 2) = 0;
                    matrixWithEigenVectors(1, 3) = primitive.u.x + a;

                    matrixWithEigenVectors(2, 0) = primitive.u.y;
                    matrixWithEigenVectors(2, 1) = primitive.u.y;
                    matrixWithEigenVectors(2, 2) = 1;
                    matrixWithEigenVectors(2, 3) = primitive.u.y;

                    matrixWithEigenVectors(3, 0) = H - primitive.u.x*a;
                    matrixWithEigenVectors(3, 1) = 0.5*primitive.u.dot(primitive.u);
                    matrixWithEigenVectors(3, 2) = primitive.u.y;
                    matrixWithEigenVectors(3, 3) = H + primitive.u.x*a;

                    return matrixWithEigenVectors;
                }
                else if (direction == 1) {
                    // We use the rotation trick, see 3.2.2 and Proposition 3.19 in Toro's book
                    // http://www.springer.com/de/book/9783540252023
                    auto conservedRotated = ConservedVariables(conserved.rho, rvec2{conserved.m.y, -conserved.m.x}, conserved.E);
                    return computeEigenVectorMatrix<0>(conservedRotated);
                }
                assert(false);
            }

            template<>
            template<int direction>
            __device__ __host__ rvec4 Euler<2>::computeEigenValues(const ConservedVariables& conserved) const {
                if (direction == 0) {
                    auto primitive = computePrimitiveVariables(conserved);
                    const real a = sqrt(gamma*primitive.p / conserved.rho);
                    return rvec4(primitive.u.x - a, primitive.u.x, primitive.u.x, primitive.u.x + a);
                }
                else if (direction == 1) {
                    // We use the rotation trick, see 3.2.2 and Proposition 3.19 in Toro's book
                    // http://www.springer.com/de/book/9783540252023
                    auto conservedRotated = ConservedVariables(conserved.rho, rvec2{conserved.m.y, -conserved.m.x}, conserved.E);
                    return computeEigenValues<0>(conservedRotated);
                }
                assert(false);
            }



            template<>
            template<int direction>
            __device__ __host__ matrix3 Euler<1>::computeEigenVectorMatrix(const ConservedVariables& conserved) const {


                if (direction == 0) {
                    matrix3 matrixWithEigenVectors;

                    auto primitive = computePrimitiveVariables(conserved);
                    const real a = sqrt(gamma*primitive.p / conserved.rho);
                    const real H = (conserved.E + primitive.p) / conserved.rho;
                    matrixWithEigenVectors(0, 0) = 1;
                    matrixWithEigenVectors(0, 1) = 1;
                    matrixWithEigenVectors(0, 3) = 1;

                    matrixWithEigenVectors(1, 0) = primitive.u.x - a;
                    matrixWithEigenVectors(1, 1) = primitive.u.x;
                    matrixWithEigenVectors(1, 3) = primitive.u.x + a;

                    matrixWithEigenVectors(2, 0) = H - primitive.u.x*a;
                    matrixWithEigenVectors(2, 1) = 0.5*primitive.u.dot(primitive.u);
                    matrixWithEigenVectors(2, 3) = H + primitive.u.x*a;

                    return matrixWithEigenVectors;
                }

                assert(false);
            }

            template<>
            template<int direction>
            __device__ __host__ rvec3 Euler<1>::computeEigenValues(const ConservedVariables& conserved) const {
                if (direction == 0) {
                    auto primitive = computePrimitiveVariables(conserved);
                    const real a = sqrt(gamma*primitive.p / conserved.rho);
                    return rvec3(primitive.u.x - a, primitive.u.x, primitive.u.x + a);
                }

                assert(false);
            }
        } 
} // namespace alsfvm
} // namespace equation

