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

#include "alsfvm/numflux/NumericalFluxCPU.hpp"
#include "alsfvm/numflux/numerical_flux_list.hpp"
#include "alsfvm/numflux/numflux_util.hpp"
#include <cassert>
#include "alsfvm/numflux/numflux_util.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include <fstream>

#include <iostream>

namespace alsfvm {
namespace numflux {


template<class Flux, class Equation, size_t direction>
void computeNetFlux(const Equation& eq, const volume::Volume& left,
    const volume::Volume& right,
    volume::Volume& out,
    volume::Volume& temporaryVolume,
    real& waveSpeed, size_t numberOfGhostCells,
    const ivec3 start, const ivec3 end) {
    // We will automate the creation of these pointer arrays soon,
    // for now we keep them to keep things simple.

    typename Equation::ConstViews leftViews(left);


    typename Equation::ConstViews rightViews(right);

    typename Equation::Views outViews(out);

    typename Equation::Views temporaryViews(temporaryVolume);

    const bool xDir = direction == 0;
    const bool yDir = direction == 1;
    const bool zDir = direction == 2;

    const int nx = out.getTotalNumberOfXCells();
    const int ny = out.getTotalNumberOfYCells();
    const int nz = out.getTotalNumberOfZCells();
    const int ngx = out.getNumberOfXGhostCells();
    const int ngy = out.getNumberOfYGhostCells();
    const int ngz = out.getNumberOfZGhostCells();

    real waveSpeedComputed = 0;
    auto stencil = getStencil<Flux>(Flux());

    for (int z = ngz - zDir + start.z; z < nz - ngz + end.z; ++z) {
        #pragma omp parallel for reduction(max: waveSpeedComputed)

        for (int y = ngy - yDir + start.y; y < ny - ngy + end.y; ++y) {
            for (int x = int(ngx) - xDir + start.x; x < int(nx) - int(ngx) + end.x; ++x) {

                // Now we need to build up the stencil for this set of indices
                decltype(stencil) indices;

                for (size_t index = 0; index < stencil.size(); ++index) {

                    indices[index] = outViews.index(
                            x + xDir * stencil[index],
                            y + yDir * stencil[index],
                            z + zDir * stencil[index]);
                }

                typename Equation::ConservedVariables flux;
                const real waveSpeedLocal = computeFluxForStencil<Flux, Equation, direction>(
                        eq,
                        indices,
                        leftViews,
                        rightViews,
                        flux);
                auto outIndex = outViews.index(x, y, z);
                eq.setViewAt(temporaryViews, outIndex, (-1.0)*flux);
                waveSpeedComputed = std::max(waveSpeedComputed, waveSpeedLocal);
            }
        }
    }

    waveSpeed = waveSpeedComputed;

    for (int z = ngz - zDir + start.z; z < nz - ngz + end.z - zDir; ++z) {
        #pragma omp parallel for

        for (int y = ngy - yDir + start.y; y < ny - ngy + end.y - yDir; ++y) {

            for (int x = int(ngx) - xDir + start.x; x < int(nx - ngx) + end.x - xDir; ++x) {

                const size_t rightIndex = outViews.index(x + xDir, y + yDir, z + zDir);
                const size_t middleIndex = outViews.index(x, y, z);
                auto fluxMiddleRight = eq.fetchConservedVariables(temporaryViews, rightIndex);
                auto fluxLeftMiddle = (-1.0) * eq.fetchConservedVariables(temporaryViews,
                        middleIndex);

                eq.addToViewAt(outViews, rightIndex, fluxMiddleRight + fluxLeftMiddle);

            }
        }
    }
}


template<class Equation>
void makeZero(Equation& equation, volume::Volume& out,
    const ivec3 start, const ivec3 end) {

    const int nx = out.getTotalNumberOfXCells();
    const int ny = out.getTotalNumberOfYCells();
    const int nz = out.getTotalNumberOfZCells();
    const int ngx = out.getNumberOfXGhostCells();
    const int ngy = out.getNumberOfYGhostCells();
    const int ngz = out.getNumberOfZGhostCells();

    typename Equation::Views outViews(out);

    for (int z = ngz + start.z; z < nz - ngz + end.z; ++z) {
        #pragma omp parallel for

        for (int y = ngy + start.y; y < ny - ngy + end.y; ++y) {

            for (int x = int(ngx) + start.x; x < int(nx - ngx) + end.x; ++x) {
                typename Equation::ConservedVariables zero;
                equation.setViewAt(outViews, outViews.index(x, y, z), zero);
            }
        }
    }

}

template<class Flux, class Equation, size_t dimension>
NumericalFluxCPU<Flux, Equation, dimension>::NumericalFluxCPU(
    const grid::Grid& grid,
    alsfvm::shared_ptr<reconstruction::Reconstruction>& reconstruction,
    const alsfvm::shared_ptr<simulator::SimulatorParameters>& simulatorParameters,
    alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration)
  : volumeFactory(Equation::getName(),
          alsfvm::make_shared<memory::MemoryFactory>(deviceConfiguration)),
      reconstruction(reconstruction),
      parameters(static_cast<typename Equation::Parameters&>
          (simulatorParameters->getEquationParameters())) {
    static_assert(dimension > 0, "We only support positive dimension!");
    static_assert(dimension < 4, "We only support dimension up to 3");


    createVolumes(grid.getDimensions().x,
        grid.getDimensions().y,
        grid.getDimensions().z,
        getNumberOfGhostCells());

}

template<class Flux, class Equation, size_t dimension>
void NumericalFluxCPU<Flux, Equation, dimension>::computeFlux(
    const volume::Volume& conservedVariables,
    rvec3& waveSpeed, bool computeWaveSpeed,
    volume::Volume& output, const ivec3& start,
    const ivec3& end
) {
    Equation eq(parameters);
    static_assert(dimension > 0, "We only support positive dimension!");
    static_assert(dimension < 4, "We only support dimension up to 3");

    // Make sure we have the correct size.
    if (conservedVariables.getTotalNumberOfXCells() !=
        left->getTotalNumberOfXCells()
        || conservedVariables.getTotalNumberOfYCells() != left->getTotalNumberOfYCells()
        || conservedVariables.getTotalNumberOfZCells() !=
        left->getTotalNumberOfZCells()) {

        createVolumes(conservedVariables.getNumberOfXCells(),
            conservedVariables.getNumberOfYCells(),
            conservedVariables.getNumberOfZCells(),
            conservedVariables.getNumberOfXGhostCells());
    }

    makeZero(eq, output, start, end);

    reconstruction->performReconstruction(conservedVariables, 0, 0,
        *left, *right, start, end);
    computeNetFlux<Flux, Equation, 0>(eq, *left, *right, output,
        *temporaryVolume, waveSpeed.x,
        getNumberOfGhostCells(), start, end);

    if (dimension > 1) {
        reconstruction->performReconstruction(conservedVariables, 1, 0,
            *left, *right, start, end);
        computeNetFlux<Flux, Equation, 1>(eq, *left, *right, output,
            *temporaryVolume,
            waveSpeed.y, getNumberOfGhostCells(),
            start, end);
    }


    if (dimension > 2) {
        reconstruction->performReconstruction(conservedVariables, 2, 0,
            *left, *right, start, end);
        computeNetFlux<Flux, Equation, 2>(eq, *left, *right, output,
            *temporaryVolume, waveSpeed.z,
            getNumberOfGhostCells(),
            start, end);
    }

}

///
/// \returns the number of ghost cells this specific flux requires
///
template<class Flux, class Equation, size_t dimension>
size_t NumericalFluxCPU<Flux, Equation, dimension>::getNumberOfGhostCells() {
    return std::max(getStencil<Flux>(Flux()).size() - 1,
            reconstruction->getNumberOfGhostCells());

}

template<class Flux, class Equation, size_t dimension>
void NumericalFluxCPU<Flux, Equation, dimension>::createVolumes(size_t nx,
    size_t ny, size_t nz, size_t ngc) {
    left = volumeFactory.createConservedVolume(nx,
            ny,
            nz,
            ngc);
    left->makeZero();

    right = volumeFactory.createConservedVolume(nx,
            ny,
            nz,
            ngc);

    right->makeZero();

    temporaryVolume = volumeFactory.createConservedVolume(nx,
            ny,
            nz,
            ngc);
}

ALSFVM_FLUX_INSTANTIATE(NumericalFluxCPU)
}
}
