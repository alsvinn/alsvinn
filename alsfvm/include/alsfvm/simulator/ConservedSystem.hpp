#pragma once
#include "alsfvm/integrator/System.hpp"
#include "alsfvm/numflux/NumericalFlux.hpp"

namespace alsfvm { namespace simulator { 

    /// 
    class ConservedSystem : public integrator::System {
    public:
        ConservedSystem(alsfvm::shared_ptr<numflux::NumericalFlux>& numericalFlux);
        
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
                                volume::Volume& output);

    private:
        alsfvm::shared_ptr<numflux::NumericalFlux> numericalFlux;
    };
} // namespace alsfvm
} // namespace simulator
