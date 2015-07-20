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
                     const size_t cellScaling,
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
        const int z = z;

        // Right indices
        const int xr = i + (direction == 0);
        const int yr = j + (direction == 1);
        const int zr = k + (direction == 2);


        // This needs to be done with some smart template recursion
        const size_t indexLeft = index(xl, yl, xl);
        equation::euler::AllVariables left = makeVariableStruct<AllVariables>(
                conservedVariables[0][indexLeft],
                conservedVariables[1][indexLeft],
                conservedVariables[2][indexLeft],
                conservedVariables[3][indexLeft],
                conservedVariables[4][indexLeft],
                extraVariables    [0][indexLeft],
                extraVariables    [1][indexLeft],
                extraVariables    [2][indexLeft],
                extraVariables    [3][indexLeft]
                );

        const size_t indexRight = index(xr, yr, xr);
        equation::euler::AllVariables right = makeVariableStruct<AllVariables>(
                conservedVariables[0][indexRight],
                conservedVariables[1][indexRight],
                conservedVariables[2][indexRight],
                conservedVariables[3][indexRight],
                conservedVariables[4][indexRight],
                extraVariables    [0][indexRight],
                extraVariables    [1][indexRight],
                extraVariables    [2][indexRight],
                extraVariables    [3][indexRight]
                );

        const size_t indexMiddle = index(x, y, z);
        equation::euler::AllVariables middle = makeVariableStruct<AllVariables>(
                conservedVariables[0][indexMiddle],
                conservedVariables[1][indexMiddle],
                conservedVariables[2][indexMiddle],
                conservedVariables[3][indexMiddle],
                conservedVariables[4][indexMiddle],
                extraVariables    [0][indexMiddle],
                extraVariables    [1][indexMiddle],
                extraVariables    [2][indexMiddle],
                extraVariables    [3][indexMiddle]
                );


        // F(U_j, U_r)
        equation::euler::ConservedVariables fluxMiddleRight;
        Flux::template computeFlux<direction>(middle, right, fluxMiddleRight);


        equation::euler::ConservedVariables fluxLeftMiddle;
        Flux::template computeFlux<direction>(left, middle, fluxLeftMiddle);

        out = out - cellScaling*(fluxLeftMiddle - fluxMiddleRight);

    }

    template<class Flux>
    NumericalFluxCPU<Flux>::NumericalFluxCPU(const grid::Grid &grid, const std::shared_ptr<DeviceConfiguration> &deviceConfiguration)
    {
        // Empty
    }

    template<class Flux>
    void NumericalFluxCPU<Flux>::computeFlux(const volume::Volume& conservedVariables,
        const volume::Volume& extraVariables,
		const rvec3& cellLengths,
		volume::Volume& output
		) 
	{

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

        std::array<const real*, 5> conservedPointers = {
            conservedVariables.getScalarMemoryArea(0)->getPointer(),
            conservedVariables.getScalarMemoryArea(1)->getPointer(),
            conservedVariables.getScalarMemoryArea(2)->getPointer(),
            conservedVariables.getScalarMemoryArea(3)->getPointer(),
            conservedVariables.getScalarMemoryArea(4)->getPointer(),
        };


        std::array<const real*, 4> extraPointers = {
            extraVariables.getScalarMemoryArea(0)->getPointer(),
            extraVariables.getScalarMemoryArea(1)->getPointer(),
            extraVariables.getScalarMemoryArea(2)->getPointer(),
            extraVariables.getScalarMemoryArea(3)->getPointer()
        };

		for (size_t k = 1; k < nz - 1; k++) {
			for (size_t j = 1; j < ny - 1; j++) {
				for (size_t i = 1; i < nx - 1; i++) {
					equation::euler::ConservedVariables flux;

                    addFluxDirection<0, Flux>(i, j, k, conservedPointers, extraPointers, flux, index, cellLengths[0], flux);
                    addFluxDirection<1, Flux>(i, j, k, conservedPointers, extraPointers, flux, index, cellLengths[1], flux);
                    addFluxDirection<2, Flux>(i, j, k, conservedPointers, extraPointers, flux, index, cellLengths[2], flux);

				}
			}
		}
	}

	/// 
	/// \returns the number of ghost cells this specific flux requires
	///
	template<class Flux>
    size_t NumericalFluxCPU<Flux>::getNumberOfGhostCells() {
		return 1;

    }

    template class NumericalFluxCPU<HLL>;
}
}
}
