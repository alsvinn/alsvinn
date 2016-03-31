#include "alsfvm/numflux/NumericalFluxCPU.hpp"
#include "alsfvm/numflux/numerical_flux_list.hpp"
#include <cassert>
#include "alsfvm/numflux/numflux_util.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include <fstream>
#include <omp.h>
#include <iostream>

namespace alsfvm { namespace numflux {

    template<class Flux, class Equation, int direction>
    real computeNetFlux(const Equation& eq,
                  size_t middleIndex,
                  size_t rightIndex,
                  typename Equation::ConstViews & left,
                  typename Equation::ConstViews & right,
                  typename Equation::ConservedVariables& out)
    {

        const ivec3 directionVector(direction == 0, direction==1, direction==2);
        // This needs to be done with some smart template recursion

        // This is the value for j+1/2
        typename Equation::AllVariables rightJpHf = eq.fetchAllVariables(left, rightIndex);


        // This is the value for j+1/2
        typename Equation::AllVariables leftJpHf = eq.fetchAllVariables(right, middleIndex);

        // F(U_l, U_r)
        typename Equation::ConservedVariables fluxMiddleRight;
        real waveSpeed = Flux::template computeFlux<direction>(eq, leftJpHf, rightJpHf, fluxMiddleRight);

        out = fluxMiddleRight;
		return waveSpeed;
    }

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

        std::vector<real> waveSpeeds(omp_get_max_threads(), 0);
        for(size_t z = ngz - zDir; z < nz - ngz; ++z) {
            for(size_t y = ngy - yDir; y < ny - ngy; ++y) {
#pragma omp parallel for
                for(int x = ngx - xDir; x < int(nx) - int(ngx); ++x) {
                    const auto threadId = omp_get_thread_num();
                    const size_t rightIndex = outViews.index(x+xDir, y+yDir, z+zDir);
                    const size_t middleIndex = outViews.index(x, y, z);


                    typename Equation::ConservedVariables flux;
                    const real waveSpeedLocal = computeNetFlux<Flux, Equation, direction>(
                                eq,
                                middleIndex,
                                rightIndex,
                                leftViews,
                                rightViews,
                                flux);

                    eq.setViewAt(temporaryViews, middleIndex, (-1.0)*flux);
                    waveSpeeds[threadId] = std::max(waveSpeeds[threadId], waveSpeedLocal);
                }
            }
        }
        waveSpeed = *std::max_element(waveSpeeds.begin(), waveSpeeds.end());


        for(size_t z = ngz - zDir; z < nz - ngz; ++z) {
            for(size_t y = ngy - yDir; y < ny - ngy; ++y) {
#pragma omp parallel for
                for(int x = ngx - xDir; x < int(nx - ngx); ++x) {

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
        : reconstruction(reconstruction), parameters(static_cast<typename Equation::Parameters&>(simulatorParameters->getEquationParameters()))
    {
        static_assert(dimension > 0, "We only support positive dimension!");
        static_assert(dimension < 4, "We only support dimension up to 3");

        alsfvm::shared_ptr<memory::MemoryFactory> memoryFactory(new memory::MemoryFactory(deviceConfiguration));
        volume::VolumeFactory volumeFactory(Equation::name, memoryFactory);

        left = volumeFactory.createConservedVolume(grid.getDimensions().x,
                                                   grid.getDimensions().y,
                                                   grid.getDimensions().z,
                                                   getNumberOfGhostCells());
        left->makeZero();

        right = volumeFactory.createConservedVolume(grid.getDimensions().x,
                                                   grid.getDimensions().y,
                                                   grid.getDimensions().z,
                                                    getNumberOfGhostCells());

        right->makeZero();

        temporaryVolume = volumeFactory.createConservedVolume(grid.getDimensions().x,
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
        return reconstruction->getNumberOfGhostCells();

    }

    ALSFVM_FLUX_INSTANTIATE(NumericalFluxCPU)
}
}
