#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/equation/EquationParameters.hpp"
#include "alsfvm/equation/burgers/ConservedVariables.hpp"
#include "alsfvm/equation/burgers/ExtraVariables.hpp"
#include "alsfvm/equation/burgers/AllVariables.hpp"
#include "alsfvm/equation/burgers/PrimitiveVariables.hpp"

#include "alsfvm/equation/burgers/Views.hpp"
#include "alsfvm/volume/Volume.hpp"
#include "alsfvm/equation/burgers/ViewsExtra.hpp"

namespace alsfvm { namespace equation { namespace burgers { 

class Burgers {
public:

    Burgers(const EquationParameters& parameters)
    {
        // empty
    }

    typedef EquationParameters Parameters;
    typedef burgers::ConservedVariables ConservedVariables;
    typedef burgers::ExtraVariables ExtraVariables;
    typedef burgers::PrimitiveVariables PrimitiveVariables;
    typedef burgers::AllVariables AllVariables;

    ///
    /// Defaults to "burgers".
    ///
    static const std::string name;

    //!
    //! List of all conserved variables used by Burgers (u)
    //!
    static const std::vector<std::string> conservedVariables;

    //!
    //! List of all primtive variables used by Burgers (u)
    //!
    static const std::vector<std::string> primitiveVariables;

    //!
    //! List of all extra variables used by Burgers (none)
    //!
    static const std::vector<std::string> extraVariables;

    ///
    /// Gives the number of conserved variables used (1)
    ///
    static const size_t  numberOfConservedVariables = 1;

    __device__ __host__ static size_t getNumberOfConservedVariables() {
        return 1;
    }

    typedef equation::burgers::Views<volume::Volume, memory::View<real> > Views;
    typedef equation::burgers::Views<const volume::Volume, const memory::View<const real> > ConstViews;

    typedef equation::burgers::ViewsExtra<volume::Volume, memory::View<real> > ViewsExtra;
    typedef equation::burgers::ViewsExtra<const volume::Volume, const memory::View<const real> > ConstViewsExtra;


    ///
    /// Fetches and computes the all variables from memory
    ///
    __device__ __host__ AllVariables fetchAllVariables(ConstViews& views, size_t index) const {
        return makeAllVariables(views.u.at(index));
    }

    template<class T, class S>
    __device__ __host__ static  ConservedVariables fetchConservedVariables(burgers::Views<T, S>& views, size_t index)  {
        return ConservedVariables(views.u.at(index));
    }

    __device__ __host__ ExtraVariables fetchExtraVariables(ConstViewsExtra& views, size_t index) const {
        return ExtraVariables();
    }

    ///
    /// Writes the ConservedVariable struct back to memory
    ///
    __device__ __host__ static void setViewAt(Views& output, size_t index, const ConservedVariables& input)   {
        output.u.at(index) = input.u;
    }

    ///
    /// Writes the ExtraVariable struct back to memory
    ///
    __device__ __host__ void setExtraViewAt(ViewsExtra& output, size_t index, const ExtraVariables& input) const {
        // empty
    }

    ///
    /// Adds the conserved variables to the view at the given index
    ///
    /// Basically sets output[index] += input
    ///
    __device__ __host__ static void addToViewAt(Views& output, size_t index, const ConservedVariables& input)  {
        output.u.at(index) += input.u;
    }

    ///
    /// Computes the point flux.
    ///
    /// Here we view the Burgers equation as the following hyperbolic system
    /// \f[u_t+\left(\frac{u^2}{2}\right)_x=0,\f]
    ///
    /// whence the function will return \f$u^2/2\f$
    ///
    /// \param[in] u the variables to use
    /// \param[out] F the resulting flux
    ///

    template<size_t direction>
    __device__ __host__  void computePointFlux(const AllVariables& u, ConservedVariables& F) const {
        static_assert(direction < 3, "We only support up to three dimensions");

        F = u.u * u.u / 2;
    }



    ///
    /// Empty function, Burgers has no extra variables at the moment
    ///
    __device__ __host__  ExtraVariables computeExtra(const ConservedVariables& u) const {
        return ExtraVariables();
    }

    ///
    /// \brief computes the extra variables from the primitive ones
    /// \note Empty function, Burgers has no extra varaibles at the moment.
    ///
    __device__ __host__ ExtraVariables computeExtra(const PrimitiveVariables& primitiveVariables) const {
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
    __device__ __host__ ConservedVariables computeConserved(const PrimitiveVariables& primitiveVariables) const {
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

        return fabs(u.u);
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

        return u.u < INFINITY && (u.u == u.u);
    }

    __device__ __host__ AllVariables makeAllVariables(real u) const {

        return AllVariables(u);
    }

    __device__ __host__ real getWeight(const ConstViews& in, size_t index) const {
        return in.u.at(index);
    }

    __device__ __host__ PrimitiveVariables computePrimitiveVariables(const ConservedVariables& conserved) const {
        return PrimitiveVariables(conserved.u);
    }

    __device__ __host__ rvec1 computeEntropyVariables(const ConservedVariables& conserved) const {
       
        return conserved.u == 0 ? 0 : rvec1(2 / (conserved.u * (conserved.u - 2)));
    }

    __device__ __host__ rvec1 computeEntropyVariablesMultipliedByEigenVectorMatrix(const ConservedVariables& conserved) const {
        return rvec1(2.0 / (conserved.u - 2));
    }

    __device__ __host__ matrix1 computeEigenVectorMatrix(const ConservedVariables& conserved) const {
        
        matrix1 matrixWithEigenVectors;
        matrixWithEigenVectors(0, 0) = 1;
        return matrixWithEigenVectors;
    }

    __device__ __host__ rvec1 computeEigenValues(const ConservedVariables& conserved) const {
        return conserved.u;
    }

};
} // namespace alsfvm
                                      } // namespace equation
                 } // namespace burgers
