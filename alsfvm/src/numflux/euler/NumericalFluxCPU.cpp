#include "alsfvm/numflux/euler/NumericalFluxCPU.hpp"
#include "alsfvm/numflux/euler/HLL.hpp"
#include <cassert>
#include "alsfvm/numflux/numflux_util.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"

namespace alsfvm { namespace numflux { namespace euler { 

    template<class Flux, int direction>
    void computeNetFlux(size_t indexLeft,
                  size_t indexMiddle,
                  size_t indexRight,
                  const std::array<const real*, 5>& left,
                  const std::array<const real*, 5>& right,
                  const real& cellLength,
                  equation::euler::ConservedVariables& out)
    {

        // This needs to be done with some smart template recursion

        // This is the value for j+1/2
        equation::euler::AllVariables rightJpHf = equation::euler::Euler::makeAllVariables(
                left[0][indexRight],
                left[1][indexRight],
                left[2][indexRight],
                left[3][indexRight],
                left[4][indexRight]
                );


        // This is the value for j+1/2
        equation::euler::AllVariables leftJpHf = equation::euler::Euler::makeAllVariables(
                right[0][indexMiddle],
                right[1][indexMiddle],
                right[2][indexMiddle],
                right[3][indexMiddle],
                right[4][indexMiddle]
                );


        // This is the valuefor j-1/2
        equation::euler::AllVariables leftJmHf = equation::euler::Euler::makeAllVariables(
                right[0][indexLeft],
                right[1][indexLeft],
                right[2][indexLeft],
                right[3][indexLeft],
                right[4][indexLeft]
                );


        // This is the valuefor j-1/2
        equation::euler::AllVariables rightJmHf = equation::euler::Euler::makeAllVariables(
                left[0][indexMiddle],
                left[1][indexMiddle],
                left[2][indexMiddle],
                left[3][indexMiddle],
                left[4][indexMiddle]
                );



        // F(U_j, U_r)
        equation::euler::ConservedVariables fluxMiddleRight;
        Flux::template computeFlux<direction>(leftJpHf, rightJpHf, fluxMiddleRight);


        equation::euler::ConservedVariables fluxLeftMiddle;
        Flux::template computeFlux<direction>(leftJmHf, rightJmHf, fluxLeftMiddle);

        out = cellLength*(fluxLeftMiddle - fluxMiddleRight);
    }

    template<class Flux, size_t direction>
    void computeNetFlux(const volume::Volume& left, const volume::Volume& right,
                        volume::Volume& out, real cellLength, size_t numberOfGhostCells) {
        // We will automate the creation of these pointer arrays soon,
        // for now we keep them to keep things simple.
        std::array<const real*, 5> leftPointers = {
            left.getScalarMemoryArea(0)->getPointer(),
            left.getScalarMemoryArea(1)->getPointer(),
            left.getScalarMemoryArea(2)->getPointer(),
            left.getScalarMemoryArea(3)->getPointer(),
            left.getScalarMemoryArea(4)->getPointer()
        };

        std::array<const real*, 5> rightPointers = {
            right.getScalarMemoryArea(0)->getPointer(),
            right.getScalarMemoryArea(1)->getPointer(),
            right.getScalarMemoryArea(2)->getPointer(),
            right.getScalarMemoryArea(3)->getPointer(),
            right.getScalarMemoryArea(4)->getPointer()
        };


        std::array<real*, 5> outPointers = {
            out.getScalarMemoryArea(0)->getPointer(),
            out.getScalarMemoryArea(1)->getPointer(),
            out.getScalarMemoryArea(2)->getPointer(),
            out.getScalarMemoryArea(3)->getPointer(),
            out.getScalarMemoryArea(4)->getPointer()
        };

        volume::for_each_internal_volume_index(out, direction,
                                               [&](size_t leftIndex, size_t middleIndex, size_t rightIndex) {

            equation::euler::ConservedVariables flux;
            computeNetFlux<Flux, direction>(leftIndex,
                                 middleIndex,
                                 rightIndex,
                                 leftPointers,
                                 rightPointers,
                                 cellLength,
                                 flux);

            outPointers[0][middleIndex] += flux.rho;
            outPointers[1][middleIndex] += flux.m.x;
            outPointers[2][middleIndex] += flux.m.y;
            outPointers[3][middleIndex] += flux.m.z;
            outPointers[4][middleIndex] += flux.E;

            assert(!std::isnan(flux.rho));
            assert(!std::isnan(flux.m.x));
            assert(!std::isnan(flux.m.y));
            assert(!std::isnan(flux.m.z));
            assert(!std::isnan(flux.E));


        });
    }


    template<class Flux, size_t dimension>
    NumericalFluxCPU<Flux, dimension>::NumericalFluxCPU(const grid::Grid &grid,
                                                        std::shared_ptr<reconstruction::Reconstruction>& reconstruction,
                                                        std::shared_ptr<DeviceConfiguration> &deviceConfiguration)
        : reconstruction(reconstruction)
    {
        static_assert(dimension > 0, "We only support positive dimension!");
        static_assert(dimension < 4, "We only support dimension up to 3");

        std::shared_ptr<memory::MemoryFactory> memoryFactory(new memory::MemoryFactory(deviceConfiguration));
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
    }

    template<class Flux, size_t dimension>
    void NumericalFluxCPU<Flux, dimension>::computeFlux(const volume::Volume& conservedVariables,
		const rvec3& cellLengths,
		volume::Volume& output
		) 
	{
        static_assert(dimension > 0, "We only support positive dimension!");
        static_assert(dimension < 4, "We only support dimension up to 3");

        output.makeZero();


        reconstruction->performReconstruction(conservedVariables, 0, 0, *left, *right);
        computeNetFlux<Flux, 0>(*left, *right, output, cellLengths.x, getNumberOfGhostCells());

        if (dimension > 1) {
            reconstruction->performReconstruction(conservedVariables, 1, 0, *left, *right);
            computeNetFlux<Flux, 1>(*left, *right, output, cellLengths.y, getNumberOfGhostCells());
        }
        if (dimension > 2) {
            reconstruction->performReconstruction(conservedVariables, 2, 0, *left, *right);
            computeNetFlux<Flux, 2>(*left, *right, output, cellLengths.z, getNumberOfGhostCells());
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
}
}
}
