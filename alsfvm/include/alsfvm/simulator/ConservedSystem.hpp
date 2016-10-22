#pragma once
#include "alsfvm/integrator/System.hpp"
#include "alsfvm/numflux/NumericalFlux.hpp"
#include "alsfvm/diffusion/NoDiffusion.hpp"

namespace alsfvm { namespace simulator { 

    /// 
    class ConservedSystem : public integrator::System {
    public:
        ConservedSystem(alsfvm::shared_ptr<numflux::NumericalFlux>& numericalFlux,
                        alsfvm::shared_ptr<diffusion::DiffusionOperator>& diffusionOperator);
        
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

        /// 
        /// Returns the number of ghost cells needed.
        /// This will take the maximum between the number of ghost cells the numerical
        /// flux needs, and the number of ghost cells the diffusion operator needs
        ///
        virtual size_t getNumberOfGhostCells() const ;
    private:
        alsfvm::shared_ptr<numflux::NumericalFlux> numericalFlux;
        alsfvm::shared_ptr<diffusion::DiffusionOperator> diffusionOperator;
    };
} // namespace alsfvm
} // namespace simulator
