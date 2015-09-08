#include "alsfvm/numflux/euler/NumericalFluxCPU.hpp"
#include "alsfvm/numflux/euler/HLL.hpp"
#include <cassert>
#include "alsfvm/numflux/numflux_util.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"

namespace alsfvm { namespace numflux { namespace euler { 

    template<class Flux, int direction>
    void computeNetFlux(size_t leftIndex,
                  size_t middleIndex,
                  size_t rightIndex,
                  const std::array<memory::View<const real>, 5>& left,
                  const std::array<memory::View<const real>, 5>& right,
                  const real& cellLength,
                  equation::euler::ConservedVariables& out)
    {

        const ivec3 directionVector(direction == 0, direction==1, direction==2);
        // This needs to be done with some smart template recursion

        // This is the value for j+1/2
        equation::euler::AllVariables rightJpHf = equation::euler::Euler::makeAllVariables(
                left[0].at(rightIndex),
                left[1].at(rightIndex),
                left[2].at(rightIndex),
                left[3].at(rightIndex),
                left[4].at(rightIndex)
                );


        // This is the value for j+1/2
        equation::euler::AllVariables leftJpHf = equation::euler::Euler::makeAllVariables(
                right[0].at(middleIndex),
                right[1].at(middleIndex),
                right[2].at(middleIndex),
                right[3].at(middleIndex),
                right[4].at(middleIndex)
                );


#if 0
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
#endif



        // F(U_j, U_r)
        equation::euler::ConservedVariables fluxMiddleRight;
        Flux::template computeFlux<direction>(leftJpHf, rightJpHf, fluxMiddleRight);
#if 0

        equation::euler::ConservedVariables fluxLeftMiddle;
        Flux::template computeFlux<direction>(leftJmHf, rightJmHf, fluxLeftMiddle);

        out = cellLength*(fluxLeftMiddle) - cellLength*(fluxMiddleRight);
#else
        out = cellLength*(fluxMiddleRight);
#endif
    }

    template<class Flux, size_t direction>
    void computeNetFlux(const volume::Volume& left, const volume::Volume& right,
                        volume::Volume& out, real cellLength, size_t numberOfGhostCells) {
        // We will automate the creation of these pointer arrays soon,
        // for now we keep them to keep things simple.
        std::array<memory::View<const real>, 5> leftViews = {
            left.getScalarMemoryArea(0)->getView(),
            left.getScalarMemoryArea(1)->getView(),
            left.getScalarMemoryArea(2)->getView(),
            left.getScalarMemoryArea(3)->getView(),
            left.getScalarMemoryArea(4)->getView()
        };

        std::array<memory::View<const real>, 5> rightViews = {
            right.getScalarMemoryArea(0)->getView(),
            right.getScalarMemoryArea(1)->getView(),
            right.getScalarMemoryArea(2)->getView(),
            right.getScalarMemoryArea(3)->getView(),
            right.getScalarMemoryArea(4)->getView()
        };


        std::array<memory::View<real>, 5> outViews = {
            out.getScalarMemoryArea(0)->getView(),
            out.getScalarMemoryArea(1)->getView(),
            out.getScalarMemoryArea(2)->getView(),
            out.getScalarMemoryArea(3)->getView(),
            out.getScalarMemoryArea(4)->getView()
        };

        ivec3 directionVector(direction==0, direction==1, direction==2);

        volume::for_each_cell_index_with_neighbours<direction>(out,
                             [&](size_t leftIndex, size_t middleIndex, size_t rightIndex) {

            equation::euler::ConservedVariables flux;
            computeNetFlux<Flux, direction>(
                                 leftIndex,
                                 middleIndex,
                                 rightIndex,
                                 leftViews,
                                 rightViews,
                                 cellLength,
                                 flux);

            outViews[0].at(middleIndex) -= flux.rho;
            outViews[1].at(middleIndex) -= flux.m.x;
            outViews[2].at(middleIndex) -= flux.m.y;
            outViews[3].at(middleIndex) -= flux.m.z;
            outViews[4].at(middleIndex) -= flux.E;

            outViews[0].at(rightIndex) += flux.rho;
            outViews[1].at(rightIndex) += flux.m.x;
            outViews[2].at(rightIndex) += flux.m.y;
            outViews[3].at(rightIndex) += flux.m.z;
            outViews[4].at(rightIndex) += flux.E;

            assert(!std::isnan(flux.rho));
            assert(!std::isnan(flux.m.x));
            assert(!std::isnan(flux.m.y));
            assert(!std::isnan(flux.m.z));
            assert(!std::isnan(flux.E));


        }, ivec3(left.getNumberOfXGhostCells(),
            left.getNumberOfYGhostCells(),
            left.getNumberOfZGhostCells())- directionVector,
           ivec3(left.getNumberOfXGhostCells(),
            left.getNumberOfYGhostCells(),
            left.getNumberOfZGhostCells()));
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
