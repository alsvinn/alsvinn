/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once
#include "alsfvm/integrator/System.hpp"
#include "alsfvm/numflux/NumericalFlux.hpp"
#include "alsfvm/diffusion/NoDiffusion.hpp"
#include "alsfvm/mpi/CellExchanger.hpp"
namespace alsfvm {
namespace simulator {

///
class ConservedSystem : public integrator::System {
public:
    ConservedSystem(alsfvm::shared_ptr<numflux::NumericalFlux> numericalFlux,
        alsfvm::shared_ptr<diffusion::DiffusionOperator> diffusionOperator);

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
    virtual void operator()( volume::Volume& conservedVariables,
        rvec3& waveSpeed, bool computeWaveSpeed,
        volume::Volume& output);

    ///
    /// Returns the number of ghost cells needed.
    /// This will take the maximum between the number of ghost cells the numerical
    /// flux needs, and the number of ghost cells the diffusion operator needs
    ///
    virtual size_t getNumberOfGhostCells() const ;

    void setCellExchanger(mpi::CellExchangerPtr cellExchanger);
private:
    alsfvm::shared_ptr<numflux::NumericalFlux> numericalFlux;
    alsfvm::shared_ptr<diffusion::DiffusionOperator> diffusionOperator;

    mpi::CellExchangerPtr cellExchanger{nullptr};
};
} // namespace alsfvm
} // namespace simulator
