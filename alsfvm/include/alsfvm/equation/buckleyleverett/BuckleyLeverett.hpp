#pragma once
#include "alsfvm/types.hpp"
#include <cmath>
#include "alsfvm/equation/EquationParameters.hpp"
#include "alsfvm/equation/buckleyleverett/ConservedVariables.hpp"
#include "alsfvm/equation/buckleyleverett/ExtraVariables.hpp"
#include "alsfvm/equation/buckleyleverett/AllVariables.hpp"
#include "alsfvm/equation/buckleyleverett/PrimitiveVariables.hpp"

#include "alsfvm/equation/buckleyleverett/Views.hpp"
#include "alsfvm/volume/Volume.hpp"
#include "alsfvm/equation/buckleyleverett/ViewsExtra.hpp"
#define USE_LOG_ENTROPY 1
namespace alsfvm {
namespace equation {
namespace buckleyleverett {

class BuckleyLeverett {
public:

    BuckleyLeverett(const EquationParameters& parameters) {
        // empty
    }

    typedef EquationParameters Parameters;
    typedef buckleyleverett::ConservedVariables ConservedVariables;
    typedef buckleyleverett::ExtraVariables ExtraVariables;
    typedef buckleyleverett::PrimitiveVariables PrimitiveVariables;
    typedef buckleyleverett::AllVariables AllVariables;

    ///
    /// Defaults to "buckleyleverett".
    ///
    static std::string getName() {
        return "buckleyleverett";
    }

    //!
    //! List of all conserved variables used by buckleyleverett (u)
    //!
    static const std::vector<std::string> conservedVariables;

    //!
    //! List of all primtive variables used by buckleyleverett (u)
    //!
    static const std::vector<std::string> primitiveVariables;

    //!
    //! List of all extra variables used by buckleyleverett (none)
    //!
    static const std::vector<std::string> extraVariables;

    ///
    /// Gives the number of conserved variables used (1)
    ///
    static const size_t  numberOfConservedVariables = 1;

    ///
    /// Gives the lower bound for the parameter the entropy functions,
    /// corresponds to the "a" variable in the tecno variable for the buckleyleverett
    /// log entropy
    ///
    static const constexpr real entropyLowerBound = 0;

    ///
    /// Gives the lower bound for the parameter the entropy functions,
    /// corresponds to the "b" variable in the tecno variable for the buckleyleverett
    /// log entropy
    ///
    static const constexpr real entropyUpperBound = 2;

    __device__ __host__ static size_t getNumberOfConservedVariables() {
        return 1;
    }

    typedef equation::buckleyleverett::Views<volume::Volume, memory::View<real> >
    Views;
    typedef equation::buckleyleverett::Views<const volume::Volume, const memory::View<const real> >
    ConstViews;

    typedef equation::buckleyleverett::ViewsExtra<volume::Volume, memory::View<real> >
    ViewsExtra;
    typedef equation::buckleyleverett::ViewsExtra<const volume::Volume, const memory::View<const real> >
    ConstViewsExtra;


    ///
    /// Fetches and computes the all variables from memory
    ///
    __device__ __host__ AllVariables fetchAllVariables(ConstViews& views,
        size_t index) const {
        return makeAllVariables(views.u.at(index));
    }

    template<class T, class S>
    __device__ __host__ static  ConservedVariables fetchConservedVariables(
        buckleyleverett::Views<T, S>& views, size_t index)  {
        return ConservedVariables(views.u.at(index));
    }

    __device__ __host__ ExtraVariables fetchExtraVariables(ConstViewsExtra& views,
        size_t index) const {
        return ExtraVariables();
    }

    ///
    /// Writes the ConservedVariable struct back to memory
    ///
    __device__ __host__ static void setViewAt(Views& output, size_t index,
        const ConservedVariables& input)   {
        output.u.at(index) = input.u;
    }

    ///
    /// Writes the ExtraVariable struct back to memory
    ///
    __device__ __host__ void setExtraViewAt(ViewsExtra& output, size_t index,
        const ExtraVariables& input) const {
        // empty
    }

    ///
    /// Adds the conserved variables to the view at the given index
    ///
    /// Basically sets output[index] += input
    ///
    __device__ __host__ static void addToViewAt(Views& output, size_t index,
        const ConservedVariables& input)  {
        output.u.at(index) += input.u;
    }

    ///
    /// Computes the point flux.
    ///
    /// Here we view the buckleyleverett equation as the following hyperbolic system
    /// \f[u_t+\left(\frac{u^2}{2}\right)_x=0,\f]
    ///
    /// whence the function will return \f$u^2/2\f$
    ///
    /// \param[in] u the variables to use
    /// \param[out] F the resulting flux
    ///

    template<size_t direction>
    __device__ __host__  void computePointFlux(const AllVariables& u,
        ConservedVariables& F) const {
        static_assert(direction < 3, "We only support up to three dimensions");
        double u2 = u.u * u.u;
        F = u2 / (u2 + (1 - u.u) * (1 - u.u));
    }



    ///
    /// Empty function, buckleyleverett has no extra variables at the moment
    ///
    __device__ __host__  ExtraVariables computeExtra(const ConservedVariables& u)
    const {
        return ExtraVariables();
    }

    ///
    /// \brief computes the extra variables from the primitive ones
    /// \note Empty function, buckleyleverett has no extra varaibles at the moment.
    ///
    __device__ __host__ ExtraVariables computeExtra(const PrimitiveVariables&
        primitiveVariables) const {
        return ExtraVariables();
    }

    ///
    /// \brief computes the conserved variables from the primitive ones
    ///
    /// \param primitiveVariables the primtive variables
    /// \return the computed all variables
    /// \note This implementation is not made for speed! Should only be
    /// used sparsely (eg. for initialization).
    ///
    __device__ __host__ ConservedVariables computeConserved(
        const PrimitiveVariables& primitiveVariables) const {
        return ConservedVariables(primitiveVariables.u);
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

        return fabs(2 * u.u * (u.u * u.u + (1 - u.u) * (1 - u.u)) - u.u * u.u *
                (2 * u.u - 2 * (1 - u.u)))
            / ((u.u * u.u + (1 - u.u) * (1 - u.u)) * (u.u * u.u + (1 - u.u) * (1 - u.u)));
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
        const ExtraVariables& v) const {

        return u.u < INFINITY && (u.u == u.u);
    }

    __device__ __host__ AllVariables makeAllVariables(real u) const {

        return AllVariables(u);
    }

    __device__ __host__ real getWeight(const ConstViews& in, size_t index) const {
        return in.u.at(index);
    }

    __device__ __host__ PrimitiveVariables computePrimitiveVariables(
        const ConservedVariables& conserved) const {
        return PrimitiveVariables(conserved.u);
    }

    ///
    /// Computes the entropy variable \f$v(u)\f$ given by
    /// \f[v(u) = \partial_u E(u)\f]
    /// corresponding to the entropy
    /// \f[E(u)-\log(b-u)-\log(u-a)\f]
    /// where \f$b\f$ is given as entropyUpperBound and \f$a\f$ is given as entropyLowerBound.
    ///
    __device__ __host__ rvec1 computeEntropyVariables(const ConservedVariables&
        conserved) const {
#if USE_LOG_ENTROPY
        return (1 / (entropyUpperBound - conserved.u)) - 1 / (conserved.u -
                entropyLowerBound);
#else
        return conserved.u;
#endif

    }

    ///
    /// Computes the entropy potential \f$\psi(u)\f$ given by
    /// \f[\psi(u) = v(u)f(u) - Q(u)\f]
    /// where \f$Q(u)\f$ is defined through
    /// \f[Q'(u) = f'(u)E'(u)\f]
    ///
    __device__ __host__ rvec1 computeEntropyPotential(const ConservedVariables&
        conserved) const {
#if USE_LOG_ENTROPY
        return computeEntropyVariables(conserved) * 0.5 * conserved.u * conserved.u -
            (-2 * conserved.u
                - entropyUpperBound * log(entropyUpperBound - conserved.u) - entropyLowerBound *
                log(conserved.u - entropyLowerBound));
#else
        return 1.0 / 6.0 * (conserved.u * conserved.u * conserved.u);
#endif

    }

    template<int direction>
    __device__ __host__ rvec1 computeEntropyVariablesMultipliedByEigenVectorMatrix(
        const ConservedVariables& conserved) const {

#if USE_LOG_ENTROPY
        return rvec1(2.0 * conserved.u / (conserved.u * (conserved.u - 2)));
#else
        return rvec1(conserved.u);
#endif
    }

    template<int direction>
    __device__ __host__ matrix1 computeEigenVectorMatrix(const ConservedVariables&
        conserved) const {

        matrix1 matrixWithEigenVectors;
        matrixWithEigenVectors(0, 0) = 1;
        return matrixWithEigenVectors;
    }

    template<int direction>
    __device__ __host__ rvec1 computeEigenValues(const ConservedVariables& u)
    const {
        return  (2 * u.u * (u.u * u.u + (1 - u.u) * (1 - u.u)) - u.u * u.u *
                (2 * u.u - 2 * (1 - u.u)))
            / ((u.u * u.u + (1 - u.u) * (1 - u.u)) * (u.u * u.u + (1 - u.u) * (1 - u.u)));
    }

};
} // namespace alsfvm
} // namespace equation
} // namespace buckleyleverett
