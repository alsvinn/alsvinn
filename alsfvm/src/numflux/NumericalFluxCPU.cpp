#include "alsfvm/numflux/NumericalFluxCPU.hpp"
#include "alsfvm/numflux/numerical_flux_list.hpp"
#include "alsfvm/numflux/numflux_util.hpp"
#include <cassert>
#include "alsfvm/numflux/numflux_util.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include <fstream>

#include <iostream>

namespace alsfvm { namespace numflux {


    template<class Flux, class Equation, size_t direction>
    void computeNetFlux(const Equation& eq, const volume::Volume& left, const volume::Volume& right,
                        volume::Volume& out,
                        volume::Volume& temporaryVolume,
                        real& waveSpeed, size_t numberOfGhostCells) {
        // We will automate the creation of these pointer arrays soon,
        // for now we keep them to keep things simple.

        typename Equation::ConstViews leftViews(left);


        typename Equation::ConstViews rightViews(right);

        typename Equation::Views outViews(out);

        typename Equation::Views temporaryViews(temporaryVolume);

        const bool xDir = direction == 0;
        const bool yDir = direction == 1;
        const bool zDir = direction == 2;

        const size_t nx = out.getTotalNumberOfXCells();
        const size_t ny = out.getTotalNumberOfYCells();
        const size_t nz = out.getTotalNumberOfZCells();
        const size_t ngx = out.getNumberOfXGhostCells();
        const size_t ngy = out.getNumberOfYGhostCells();
        const size_t ngz = out.getNumberOfZGhostCells();

    real waveSpeedComputed = 0;
    auto stencil = getStencil<Flux>(Flux());
        for(size_t z = ngz - zDir; z < nz - ngz; ++z) {
#pragma omp parallel for reduction(max: waveSpeedComputed)
            for(size_t y = ngy - yDir; y < ny - ngy; ++y) {
                for(int x = int(ngx) - xDir; x < int(nx) - int(ngx); ++x) {
                    
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
        for(size_t z = ngz - zDir; z < nz - ngz; ++z) {
#pragma omp parallel for
            for(size_t y = ngy - yDir; y < ny - ngy; ++y) {

                for(int x = int(ngx) - xDir; x < int(nx - ngx); ++x) {

		  const size_t rightIndex = outViews.index(x+xDir, y+yDir, z+zDir);
                    const size_t middleIndex = outViews.index(x, y, z);
                    auto fluxMiddleRight = eq.fetchConservedVariables(temporaryViews, rightIndex);
                    auto fluxLeftMiddle = (-1.0)*eq.fetchConservedVariables(temporaryViews, middleIndex);

                    eq.addToViewAt(outViews, rightIndex, fluxMiddleRight + fluxLeftMiddle);

                }
            }
        }
    }


    template<class Flux, class Equation, size_t dimension>
    NumericalFluxCPU<Flux, Equation, dimension>::NumericalFluxCPU(const grid::Grid &grid,
                                                        alsfvm::shared_ptr<reconstruction::Reconstruction>& reconstruction,
                                                        const alsfvm::shared_ptr<simulator::SimulatorParameters>& simulatorParameters,
                                                        alsfvm::shared_ptr<DeviceConfiguration> &deviceConfiguration)
        : volumeFactory(Equation::name, alsfvm::make_shared<memory::MemoryFactory>(deviceConfiguration)),
          reconstruction(reconstruction), parameters(static_cast<typename Equation::Parameters&>(simulatorParameters->getEquationParameters()))
    {
        static_assert(dimension > 0, "We only support positive dimension!");
        static_assert(dimension < 4, "We only support dimension up to 3");


        createVolumes(grid.getDimensions().x,
                      grid.getDimensions().y,
                      grid.getDimensions().z,
                      getNumberOfGhostCells());

    }

    template<class Flux, class Equation, size_t dimension>
    void NumericalFluxCPU<Flux, Equation, dimension>::computeFlux(const volume::Volume& conservedVariables,
		rvec3& waveSpeed, bool computeWaveSpeed,
		volume::Volume& output
		) 
	{
        Equation eq(parameters);
        static_assert(dimension > 0, "We only support positive dimension!");
        static_assert(dimension < 4, "We only support dimension up to 3");

        if (conservedVariables.getTotalNumberOfXCells()!= left->getTotalNumberOfXCells()
             || conservedVariables.getTotalNumberOfYCells() != left->getTotalNumberOfYCells()
                || conservedVariables.getTotalNumberOfZCells() != left->getTotalNumberOfZCells()) {

            createVolumes(conservedVariables.getNumberOfXCells(),
                          conservedVariables.getNumberOfYCells(),
                          conservedVariables.getNumberOfZCells(),
                          conservedVariables.getNumberOfXGhostCells());
        }

        output.makeZero();

        reconstruction->performReconstruction(conservedVariables, 0, 0, *left, *right);
        computeNetFlux<Flux, Equation, 0>(eq, *left, *right, output, *temporaryVolume, waveSpeed.x, getNumberOfGhostCells());

        if (dimension > 1) {
            reconstruction->performReconstruction(conservedVariables, 1, 0, *left, *right);
            computeNetFlux<Flux, Equation, 1>(eq, *left, *right, output, *temporaryVolume, waveSpeed.y, getNumberOfGhostCells());
        }


        if (dimension > 2) {
            reconstruction->performReconstruction(conservedVariables, 2, 0, *left, *right);
            computeNetFlux<Flux, Equation, 2>(eq, *left, *right, output, *temporaryVolume, waveSpeed.z, getNumberOfGhostCells());
        }

	}

	/// 
	/// \returns the number of ghost cells this specific flux requires
	///
    template<class Flux, class Equation, size_t dimension>
    size_t NumericalFluxCPU<Flux, Equation, dimension>::getNumberOfGhostCells() {
        return std::max(getStencil<Flux>(Flux()).size()-1, reconstruction->getNumberOfGhostCells());

    }

    template<class Flux, class Equation, size_t dimension>
    void NumericalFluxCPU<Flux, Equation, dimension>::createVolumes(size_t nx, size_t ny, size_t nz, size_t ngc)
    {
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
