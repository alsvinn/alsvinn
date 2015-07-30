#include "alsfvm/numflux/euler/NumericalFluxCPU.hpp"
#include "alsfvm/numflux/euler/HLL.hpp"
#include <cassert>
#include "alsfvm/numflux/numflux_util.hpp"

namespace alsfvm { namespace numflux { namespace euler { 

    template<class T>
    T makeVariableStruct(const real& a, const real& b, const real& c,
                         const real& d) {
        return T(a,b,c,d);
    }


    template<class T>
    T makeVariableStruct(const real& a, const real& b, const real& c,
                         const real& d, const real& e) {
        return T(a,b,c,d, e);
    }


    template<class T>
    T makeVariableStruct(const real& a, const real& b, const real& c,
                         const real& d, const real& e, const real& f) {
        return T(a,b,c,d, e, f);
    }


    template<class T>
    T makeVariableStruct(const real& a, const real& b, const real& c,
                         const real& d, const real& e, const real& f, const real& g) {
        return T(a,b,c,d, e, f, g);
    }


    template<class T>
    T makeVariableStruct(const real& a, const real& b, const real& c,
                         const real& d, const real& e, const real& f, const real& g,
                         const real& h) {
        return T(a,b,c,d, e, f, g, h);
    }

    template<class T>
    T makeVariableStruct(const real& a, const real& b, const real& c,
                         const real& d, const real& e, const real& f, const real& g,
                         const real& h, const real& i) {
        return T(a,b,c,d, e, f, g, h, i);
    }

    template<class T>
    T makeVariableStruct(const real& a, const real& b, const real& c,
                         const real& d, const real& e, const real& f, const real& g,
                         const real& h, const real& i, const real& j) {
        return T(a,b,c,d, e, f, g, h, i, j);
    }
    ///
    /// This will compute
    /// \f$\mathrm{out} = \mathrm{out} + \mathrm{cellScaling} * (F(Ul, Um)-F(Um,Ur))\f$
    ///
    /// Note we reverse the traditional order as our integrator expects it in this way.
    ///
    template<int direction, class Flux>
    void addFluxDirection(int i, int j, int k, const std::array<const real*, 5>& conservedVariables,
                     const std::array<const real*, 4>& extraVariables, equation::euler::ConservedVariables& flux,
                     const std::function<size_t(size_t,size_t,size_t)>& index,
                     const real cellScaling,
                     equation::euler::ConservedVariables& out)
    {
        using namespace equation::euler;
        // Left indices
        const int xl = i - (direction == 0);
        const int yl = j - (direction == 1);
        const int zl = k - (direction == 2);

        // Middle indices
        const int x = i;
        const int y = j;
        const int z = k;

        // Right indices
        const int xr = i + (direction == 0);
        const int yr = j + (direction == 1);
        const int zr = k + (direction == 2);



        const size_t indexLeft = index(xl, yl, zl);

        // This needs to be done with some smart template recursion
        equation::euler::AllVariables left = makeVariableStruct<AllVariables>(
                conservedVariables[0][indexLeft],
                conservedVariables[1][indexLeft],
                conservedVariables[2][indexLeft],
                conservedVariables[3][indexLeft],
                conservedVariables[4][indexLeft]);
#if 0
        ,
                extraVariables    [0][indexLeft],
                extraVariables    [1][indexLeft],
                extraVariables    [2][indexLeft],
                extraVariables    [3][indexLeft]
                );
#endif

        const size_t indexRight = index(xr, yr, zr);
        equation::euler::AllVariables right = makeVariableStruct<AllVariables>(
                conservedVariables[0][indexRight],
                conservedVariables[1][indexRight],
                conservedVariables[2][indexRight],
                conservedVariables[3][indexRight],
                conservedVariables[4][indexRight]);

        const size_t indexMiddle = index(x, y, z);
        equation::euler::AllVariables middle = makeVariableStruct<AllVariables>(
                conservedVariables[0][indexMiddle],
                conservedVariables[1][indexMiddle],
                conservedVariables[2][indexMiddle],
                conservedVariables[3][indexMiddle],
                conservedVariables[4][indexMiddle]);


        // F(U_j, U_r)
        equation::euler::ConservedVariables fluxMiddleRight;
        Flux::template computeFlux<direction>(middle, right, fluxMiddleRight);


        equation::euler::ConservedVariables fluxLeftMiddle;
        Flux::template computeFlux<direction>(left, middle, fluxLeftMiddle);

        out = out - (1.0 / cellScaling)*(fluxLeftMiddle - fluxMiddleRight);

    }

    template<class Flux, size_t dimension>
    NumericalFluxCPU<Flux, dimension>::NumericalFluxCPU(const grid::Grid &grid, const std::shared_ptr<DeviceConfiguration> &deviceConfiguration)
    {
        static_assert(dimension > 0, "We only support positive dimension!");
        static_assert(dimension < 4, "We only support dimension up to 3");

    }

    template<class Flux, size_t dimension>
    void NumericalFluxCPU<Flux, dimension>::computeFlux(const volume::Volume& conservedVariables,
        const volume::Volume& extraVariables,
		const rvec3& cellLengths,
		volume::Volume& output
		) 
	{
        static_assert(dimension > 0, "We only support positive dimension!");
        static_assert(dimension < 4, "We only support dimension up to 3");

		const int nx = conservedVariables.getNumberOfXCells();
		const int ny = conservedVariables.getNumberOfYCells();
		const int nz = conservedVariables.getNumberOfZCells();


        // We need to have this guarantee for the indexing, later we will fix this.
        assert(conservedVariables.getNumberOfXCells() == conservedVariables.getScalarMemoryArea(0)->getExtentXInBytes()/sizeof(real));
        assert(conservedVariables.getNumberOfYCells() == conservedVariables.getScalarMemoryArea(0)->getExtentYInBytes()/sizeof(real));

        // Makes it easier to index
        auto index = [nx,ny,nz](const size_t i, const size_t j, const size_t k) {
            return k*nx*ny + j *nx + i;
        };

        // We will automate the creation of these pointer arrays soon,
        // for now we keep them to keep things simple.
        std::array<const real*, 5> conservedPointers = {
            conservedVariables.getScalarMemoryArea(0)->getPointer(),
            conservedVariables.getScalarMemoryArea(1)->getPointer(),
            conservedVariables.getScalarMemoryArea(2)->getPointer(),
            conservedVariables.getScalarMemoryArea(3)->getPointer(),
            conservedVariables.getScalarMemoryArea(4)->getPointer(),
        };


        // We will automate the creation of these pointer arrays soon,
        // for now we keep them to keep things simple.
        std::array<const real*, 4> extraPointers = {
            extraVariables.getScalarMemoryArea(0)->getPointer(),
            extraVariables.getScalarMemoryArea(1)->getPointer(),
            extraVariables.getScalarMemoryArea(2)->getPointer(),
            extraVariables.getScalarMemoryArea(3)->getPointer()
        };

        // We will automate the creation of these pointer arrays soon,
        // for now we keep them to keep things simple.
		std::array<real*, 5> outputPointers = {
			output.getScalarMemoryArea(0)->getPointer(),
			output.getScalarMemoryArea(1)->getPointer(),
			output.getScalarMemoryArea(2)->getPointer(),
			output.getScalarMemoryArea(3)->getPointer(),
			output.getScalarMemoryArea(4)->getPointer(),
		};

        const int hasZDirection = dimension > 2;
        const int hasYDirection = dimension > 1;

        // Notice the start and endpoints for k and j.
        // If we have z direction, then we should only iterate on the internal
        // cells (ie. start at 1 and end at nz -1). If we do not have z direction,
        // then we start at 0 and end at nz.
        const size_t startZ = hasZDirection;
        const size_t endZ = nz - hasZDirection;

        const size_t startY = hasYDirection;
        const size_t endY = ny - hasYDirection;
        for (size_t k = startZ; k < endZ; k++) {
            for (size_t j = startY; j < endY; j++) {
				for (size_t i = 1; i < nx - 1; i++) {
					const size_t outputIndex = index(i, j, k);
					equation::euler::ConservedVariables flux;

                    addFluxDirection<0, Flux>(i, j, k, conservedPointers, extraPointers, flux, index, cellLengths[0], flux);

                    if (hasYDirection) {
                        addFluxDirection<1, Flux>(i, j, k, conservedPointers, extraPointers, flux, index, cellLengths[1], flux);
                    }
                    if (hasZDirection) {
                        addFluxDirection<2, Flux>(i, j, k, conservedPointers, extraPointers, flux, index, cellLengths[2], flux);
                    }

                    outputPointers[0][outputIndex] = -flux.rho;
                    outputPointers[1][outputIndex] = -flux.m.x;
                    outputPointers[2][outputIndex] = -flux.m.y;
                    outputPointers[3][outputIndex] = -flux.m.z;
                    outputPointers[4][outputIndex] = -flux.E;

				}
			}
		}
	}

	/// 
	/// \returns the number of ghost cells this specific flux requires
	///
    template<class Flux, size_t dimension>
    size_t NumericalFluxCPU<Flux, dimension>::getNumberOfGhostCells() {
		return 1;

    }

    template class NumericalFluxCPU<HLL, 1>;
    template class NumericalFluxCPU<HLL, 2>;
    template class NumericalFluxCPU<HLL, 3>;
}
}
}
