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
#include "alsfvm/numflux/NumericalFlux.hpp"
#include "alsfvm/grid/Grid.hpp"
#include "alsfvm/reconstruction/Reconstruction.hpp"
#include "alsfvm/simulator/SimulatorParameters.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"

namespace alsfvm {
namespace numflux {

///
/// The class to compute numerical flux on the CPU
/// The template argument Flux is used to choose the concrete flux
/// The template argument dimension is to choose the correct dimension
/// (1 up to and including 3 is supported).
///
template<class Flux, class Equation, size_t dimension>
class NumericalFluxCPU : public NumericalFlux {
public:

    NumericalFluxCPU(const grid::Grid& grid,
        alsfvm::shared_ptr<reconstruction::Reconstruction>& reconstruction,
        const alsfvm::shared_ptr<simulator::SimulatorParameters>& simulatorParameters,
        alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration
    );

    ///
    /// Computes the numerical flux at each cell.
    /// This will compute the net flux in the cell, ie.
    /// \f[
    /// \mathrm{output}_{i,j,k}=\frac{\Delta t}{\Delta x}\left(F(u_{i+1,j,k}, u_{i,j,k})-F(u_{i,j,k}, u_{i-1,j,k})\right)+
    ///                         \frac{\Delta t}{\Delta y}\left(F(u_{i,j+1,k}, u_{i,j,k})-F(u_{i,j,k}, u_{i,j-1,k})\right)+
    ///                         \frac{\Delta t}{\Delta z}\left((F(u_{i,j,k+1}, u_{i,j,k})-F(u_{i,j,k}, u_{i,j,k-1})\right)
    /// \f]
    /// \param[in] conservedVariables the conservedVariables to read from (eg. for Euler: \f$\rho,\; \vec{m},\; E\f$)
    /// \param[out] waveSpeed the maximum wave speed in each direction
    /// \param[in] computeWaveSpeed should we compute the wave speeds?
    /// \param[out] output the output to write to
    /// \param[in] start (positive) the first index to compute the flux for
    /// \param[in] end (negative) the offset to on the upper part of the grid
    /// \note this will calculate the extra variables on the fly.
    ///
    virtual void computeFlux(const volume::Volume& conservedVariables,
        rvec3& waveSpeed, bool computeWaveSpeed,
        volume::Volume& output, const ivec3& start = {0, 0, 0},
        const ivec3& end = {0, 0, 0}
    );


    ///
    /// \returns the number of ghost cells this specific flux requires
    ///
    virtual size_t getNumberOfGhostCells();

private:

    void createVolumes(size_t nx, size_t ny, size_t nz, size_t ngc);
    volume::VolumeFactory volumeFactory;
    alsfvm::shared_ptr<reconstruction::Reconstruction> reconstruction;
    alsfvm::shared_ptr<volume::Volume> left;
    alsfvm::shared_ptr<volume::Volume> right;

    // This is for storing F(U_l, U_r) at each cell
    alsfvm::shared_ptr<volume::Volume> temporaryVolume;

    typename Equation::Parameters parameters;
};

} // namespace alsfvm
} // namespace numflux

