#include "alsfvm/numflux/euler/NumericalFluxCPU.hpp"
#include "alsfvm/numflux/euler/HLL.hpp"
#include "alsfvm/numflux/euler/HLL3.hpp"
#include <cassert>
#include "alsfvm/numflux/numflux_util.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include <fstream>
#include <omp.h>
#include <iostream>

namespace alsfvm { namespace numflux { namespace euler { 

    template<class Flux, int direction>
    real computeNetFlux(
                  size_t middleIndex,
                  size_t rightIndex,
                  equation::euler::Euler::ConstViews & left,
                  equation::euler::Euler::ConstViews & right,
                  equation::euler::ConservedVariables& out)
    {

        const ivec3 directionVector(direction == 0, direction==1, direction==2);
        // This needs to be done with some smart template recursion

        // This is the value for j+1/2
        equation::euler::AllVariables rightJpHf = equation::euler::Euler::fetchAllVariables(left, rightIndex);


        // This is the value for j+1/2
        equation::euler::AllVariables leftJpHf = equation::euler::Euler::fetchAllVariables(right, middleIndex);

        // F(U_l, U_r)
        equation::euler::ConservedVariables fluxMiddleRight;
        real waveSpeed = Flux::template computeFlux<direction>(leftJpHf, rightJpHf, fluxMiddleRight);

        out = fluxMiddleRight;
		return waveSpeed;
    }

    template<class Flux, size_t direction>
    void computeNetFlux(const volume::Volume& left, const volume::Volume& right,
                        volume::Volume& out,
                        volume::Volume& temporaryVolume,
                        real& waveSpeed, size_t numberOfGhostCells) {
        // We will automate the creation of these pointer arrays soon,
        // for now we keep them to keep things simple.

        equation::euler::Euler::ConstViews leftViews(left);


        equation::euler::Euler::ConstViews rightViews(right);

        equation::euler::Euler::Views outViews(out);

        equation::euler::Euler::Views temporaryViews(temporaryVolume);

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
                for(size_t x = ngx - xDir; x < nx - ngx; ++x) {
                    const auto threadId = omp_get_thread_num();
                    const size_t rightIndex = outViews.index(x+xDir, y+yDir, z+zDir);
                    const size_t middleIndex = outViews.index(x, y, z);


                    equation::euler::ConservedVariables flux;
                    const real waveSpeedLocal = computeNetFlux<Flux, direction>(
                                middleIndex,
                                rightIndex,
                                leftViews,
                                rightViews,
                                flux);

                    equation::euler::Euler::setViewAt(temporaryViews, middleIndex, (-1.0)*flux);
                    waveSpeeds[threadId] = std::max(waveSpeeds[threadId], waveSpeedLocal);
                }
            }
        }
        waveSpeed = *std::max_element(waveSpeeds.begin(), waveSpeeds.end());


        for(size_t z = ngz - zDir; z < nz - ngz; ++z) {
            for(size_t y = ngy - yDir; y < ny - ngy; ++y) {
#pragma omp parallel for
                for(size_t x = ngx - xDir; x < nx - ngx; ++x) {

                    const size_t rightIndex = outViews.index(x+xDir, y+yDir, z+zDir);
                    const size_t middleIndex = outViews.index(x, y, z);
                    auto fluxMiddleRight = equation::euler::Euler::fetchConservedVariables(temporaryViews, rightIndex);
                    auto fluxLeftMiddle = (-1.0)*equation::euler::Euler::fetchConservedVariables(temporaryViews, middleIndex);

                    equation::euler::Euler::addToViewAt(outViews, rightIndex, fluxMiddleRight + fluxLeftMiddle);

                }
            }
        }
    }


    template<class Flux, size_t dimension>
    NumericalFluxCPU<Flux, dimension>::NumericalFluxCPU(const grid::Grid &grid,
                                                        alsfvm::shared_ptr<reconstruction::Reconstruction>& reconstruction,
                                                        alsfvm::shared_ptr<DeviceConfiguration> &deviceConfiguration)
        : reconstruction(reconstruction)
    {
        static_assert(dimension > 0, "We only support positive dimension!");
        static_assert(dimension < 4, "We only support dimension up to 3");

        alsfvm::shared_ptr<memory::MemoryFactory> memoryFactory(new memory::MemoryFactory(deviceConfiguration));
        volume::VolumeFactory volumeFactory("euler", memoryFactory);

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

    template<class Flux, size_t dimension>
	void NumericalFluxCPU<Flux, dimension>::computeFlux(const volume::Volume& conservedVariables,
		rvec3& waveSpeed, bool computeWaveSpeed,
		volume::Volume& output
		) 
	{
        static_assert(dimension > 0, "We only support positive dimension!");
        static_assert(dimension < 4, "We only support dimension up to 3");

        output.makeZero();

        reconstruction->performReconstruction(conservedVariables, 0, 0, *left, *right);
        computeNetFlux<Flux, 0>(*left, *right, output, *temporaryVolume, waveSpeed.x, getNumberOfGhostCells());

        if (dimension > 1) {
            reconstruction->performReconstruction(conservedVariables, 1, 0, *left, *right);
            computeNetFlux<Flux, 1>(*left, *right, output, *temporaryVolume, waveSpeed.y, getNumberOfGhostCells());
        }


        if (dimension > 2) {
            reconstruction->performReconstruction(conservedVariables, 2, 0, *left, *right);
            computeNetFlux<Flux, 2>(*left, *right, output, *temporaryVolume, waveSpeed.z, getNumberOfGhostCells());
        }

	}

	/// 
	/// \returns the number of ghost cells this specific flux requires
	///
    template<class Flux, size_t dimension>
    size_t NumericalFluxCPU<Flux, dimension>::getNumberOfGhostCells() {
        return reconstruction->getNumberOfGhostCells();

    }

    template class NumericalFluxCPU<HLL, 1>;
    template class NumericalFluxCPU<HLL, 2>;
    template class NumericalFluxCPU<HLL, 3>;

    template class NumericalFluxCPU<HLL3, 1>;
    template class NumericalFluxCPU<HLL3, 2>;
    template class NumericalFluxCPU<HLL3, 3>;
}
}
}
