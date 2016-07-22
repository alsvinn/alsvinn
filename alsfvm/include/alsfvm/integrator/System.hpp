#pragma once
#include "alsfvm/volume/Volume.hpp"

namespace alsfvm { namespace integrator { 

    ///
    /// Abstract base class right hand side of ODEs.
    ///
    /// We model ODEs as
    ///
    /// \f[\vec{u}'(t)=F(\vec{u}(t)).\f]
    ///
    /// The system class is responsible for computing \f$F(\vec{u}(t))\f$.
    ///
    class System {
    public:

        ///
        /// \brief operator () computes the right hand side of the ODE. (see
        ///                    class definition)
        /// \param[in] conservedVariables the current state of the conserved variables
        ///                               corresponds to \f$\vec{u}\f$.
        /// \param[out] waveSpeed at end of invocation, the maximum wavespeed
        /// \param[in] computeWaveSpeed
        /// \param[out] output will at end of invocation contain the values of
        ///                    \f$F(\vec{u})\f$
        ///
        virtual void operator()(const volume::Volume& conservedVariables,
                                rvec3& waveSpeed, bool computeWaveSpeed,
                                volume::Volume& output) = 0;

        virtual ~System() {/*empty*/}
    };
} // namespace alsfvm
} // namespace integrator