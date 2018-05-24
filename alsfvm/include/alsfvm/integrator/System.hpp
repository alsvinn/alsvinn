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
#include "alsfvm/volume/Volume.hpp"
#include "alsfvm/mpi/CellExchanger.hpp"


namespace alsfvm {
namespace integrator {

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
    virtual void operator()( volume::Volume& conservedVariables,
        rvec3& waveSpeed, bool computeWaveSpeed,
        volume::Volume& output) = 0;

    ///
    /// Returns the number of ghost cells needed.
    /// This will take the maximum between the number of ghost cells the numerical
    /// flux needs, and the number of ghost cells the diffusion operator needs
    ///
    virtual inline size_t getNumberOfGhostCells() const {
        return 0;
    }

    virtual void setCellExchanger(mpi::CellExchangerPtr cellExchanger) {}

    virtual ~System() {/*empty*/}
};
} // namespace alsfvm
} // namespace integrator
