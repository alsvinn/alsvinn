#pragma once
#include "alsfvm/volume/Volume.hpp"
#include <functional>
#include <array>
#include "alsfvm/error/Exception.hpp"
#include "alsfvm/grid/Grid.hpp"
///
/// This file contains for_each functions for volumes
///

namespace alsfvm {
	namespace volume {

		///
		/// Loops through all possible cell indexes in a cache optimal manner.
		/// Example:
		/// \code{.cpp}
		/// for_each_cell_index(someVolume, [](size_t index) {
		///     std::cout << "index = " << index;
		/// }):
		/// \endcode
		///
		template<class Function>
		inline void for_each_cell_index(const Volume& in, const Function& function) {
			const size_t nx = in.getNumberOfXCells();
			const size_t ny = in.getNumberOfYCells();
			const size_t nz = in.getNumberOfZCells();
			for (size_t k = 0; k < nz; k++) {
				for (size_t j = 0; j < ny; j++) {
                    for (size_t i = 0; i < nx; i++) {
                        size_t index = k*nx*ny + j*nx + i;
						function(index);
					}
				}
			}
		}


		template<class VariableStruct>
		inline VariableStruct expandVariableStruct(const std::array<const real*, 5>& in, size_t index) {
			return VariableStruct(in[0][index], in[1][index], in[2][index], in[3][index], in[4][index]);
		}

		template<class VariableStruct>
		inline VariableStruct expandVariableStruct(const std::array<const real*, 4>& in, size_t index) {
			return VariableStruct(in[0][index], in[1][index], in[2][index], in[3][index]);
		}

		template<class VariableStruct>
		inline void saveVariableStruct(const VariableStruct& in, size_t index, std::array<real*, 4>& out) {
			real* inAsRealPointer = (real*)&in;

			out[0][index] = inAsRealPointer[0];
			out[1][index] = inAsRealPointer[1];
			out[2][index] = inAsRealPointer[2];
			out[3][index] = inAsRealPointer[3];
		}


		template<class VariableStruct>
		inline void saveVariableStruct(const VariableStruct& in, size_t index, std::array<real*, 5>& out) {
			real* inAsRealPointer = (real*)&in;

			out[0][index] = inAsRealPointer[0];
			out[1][index] = inAsRealPointer[1];
			out[2][index] = inAsRealPointer[2];
			out[3][index] = inAsRealPointer[3];
			out[4][index] = inAsRealPointer[4];
		}


		///
		/// Loops through each cell in the index and calls function function on each value of each cell
		/// Example usage
		/// \code{.cpp}
		/// transform_volume<euler::ConservedVariables, euler::ExtraVariables>(conserved, extra, 
		/// [] (const euler::ConservedVariables& in) {
		///     return euler::Euler::computeExtra(in);
		/// });
		/// \endcode
		///
		template<class VariableStructIn, class VariableStructOut>
		inline void transform_volume(const Volume& in, Volume& out,
			const std::function<VariableStructOut(const VariableStructIn&)>& function) {

			std::array<const real*, sizeof(VariableStructIn) / sizeof(real)> pointersIn;
			for (size_t i = 0; i < in.getNumberOfVariables(); i++) {
				pointersIn[i] = in.getScalarMemoryArea(i)->getPointer();
			}

			std::array<real*, sizeof(VariableStructOut) / sizeof(real)> pointersOut;
			for (size_t i = 0; i < out.getNumberOfVariables(); i++) {
				pointersOut[i] = out.getScalarMemoryArea(i)->getPointer();
			}

			for_each_cell_index(in, [&](size_t index) {
				auto out = function(expandVariableStruct<VariableStructIn>(pointersIn, index));
				saveVariableStruct(out, index, pointersOut);

			});

		}


		/// 
		/// Loops through each cell and calls the function for each cell
		/// Example
		/// \code{.cpp}
		/// for_each_cell<euler::ConservedVariables>(conserved, [](const euler::ConservedVariables& in, size_t index) {
		///    // Do something with in or index
		/// });
		/// \endcode
		///
		template<class VariableStruct>
		inline void for_each_cell(const Volume& in,
			const std::function<void(const VariableStruct&, size_t index)>& function) {
			std::array<const real*, sizeof(VariableStruct) / sizeof(real)> pointersIn;
			if (pointersIn.size() != in.getNumberOfVariables()) {
				THROW("We expected to get " << pointersIn.size() << " variables, but got " << in.getNumberOfVariables());
			}
			for (size_t i = 0; i < in.getNumberOfVariables(); i++) {
				pointersIn[i] = in.getScalarMemoryArea(i)->getPointer();
			}

			for_each_cell_index(in, [&](size_t index) {
				function(expandVariableStruct<VariableStruct>(pointersIn, index), index);
			});
		}

		/// 
		/// Loops through each cell and calls the function for each cell
		/// Example
		/// \code{.cpp}
		/// for_each_cell<euler::ConservedVariables, euler::ExtraVariables>
		///   (conserved, extra, [](const euler::ConservedVariables& inA, const euler::ExtraVariables& inB, size_t index) {
		///    // Do something with inA, inB or index
		/// });
		/// \endcode
		///
		template<class VariableStructA, class VariableStructB>
		inline void for_each_cell(const Volume& inA, const Volume& inB,
			const std::function<void(const VariableStructA&, const VariableStructB&, size_t index)>& function) {
			std::array<const real*, sizeof(VariableStructA) / sizeof(real)> pointersInA;
			std::array<const real*, sizeof(VariableStructB) / sizeof(real)> pointersInB;
			if (pointersInA.size() != inA.getNumberOfVariables()) {
				THROW("We expected to get " << pointersInA.size() << " variables, but got " << inA.getNumberOfVariables());
			}

			if (pointersInB.size() != inB.getNumberOfVariables()) {
				THROW("We expected to get " << pointersInB.size() << " variables, but got " << inB.getNumberOfVariables());
			}


			for (size_t i = 0; i < inA.getNumberOfVariables(); i++) {
				pointersInA[i] = inA.getScalarMemoryArea(i)->getPointer();
			}

			for (size_t i = 0; i < inB.getNumberOfVariables(); i++) {
				pointersInB[i] = inB.getScalarMemoryArea(i)->getPointer();
			}

			for_each_cell_index(inA, [&](size_t index) {
				function(expandVariableStruct<VariableStructA>(pointersInA, index),
					expandVariableStruct<VariableStructB>(pointersInB, index), index);
			});
		}

		/// 
		/// Fill the volume based on a filler function (depending on position).
		/// Example (making a simple Riemann problem in 2D):
		/// \code{.cpp}
		/// fill_volume<ConservedVariables, ExtraVariables>(conserved, extra, grid,
		///    [](real x, real y, real z, ConservedVariables& outConserved, ExtraVariables& outExtra) {
		///        if ( x < 0.5) {
		///           outConserved.rho = 1.0;
		///           outConserved.m = rvec3(2,1,1);
		///           outConserved.E = 8;
		///        } else {
		///           outConserved.rho = 1.2;
		///           outConserved.m = rvec3(2,1,1);
		///           outConserved.E = 9;
		///        }
		///        outExtra = Equation::calculateExtra(outConserved);
		///    });
		/// \endcode
		///
	template<class VariableStructA, class VariableStructB>
	inline void fill_volume(Volume& outA,  Volume& outB,
		const grid::Grid& grid,
		const std::function < void(real x, real y, real z, VariableStructA& outA,
		VariableStructB& outB) > & fillerFunction) {

		std::array<real*, sizeof(VariableStructA) / sizeof(real)> pointersOutA;
		std::array<real*, sizeof(VariableStructB) / sizeof(real)> pointersOutB;
		if (pointersOutA.size() != outA.getNumberOfVariables()) {
			THROW("We expected to get " << pointersOutA.size() << " variables, but got " << outA.getNumberOfVariables());
		}

		if (pointersOutB.size() != outB.getNumberOfVariables()) {
			THROW("We expected to get " << pointersOutB.size() << " variables, but got " << outB.getNumberOfVariables());
		}


		for (size_t i = 0; i < outA.getNumberOfVariables(); i++) {
			pointersOutA[i] = outA.getScalarMemoryArea(i)->getPointer();
		}

		for (size_t i = 0; i < outB.getNumberOfVariables(); i++) {
			pointersOutB[i] = outB.getScalarMemoryArea(i)->getPointer();
		}

		const size_t nx = outA.getNumberOfXCells();
		const size_t ny = outA.getNumberOfYCells();
		const size_t nz = outA.getNumberOfZCells();

		auto& midPoints = grid.getCellMidpoints();
		for (size_t k = 0; k < nz; k++) {
			for (size_t j = 0; j < ny; j++) {
				for (size_t i = 0; i < nx; i++) {
                    size_t index = k*nx*ny + j*nx + i;
					auto midPoint = midPoints[index];
					VariableStructA a;
					VariableStructB b;
					fillerFunction(midPoint.x, midPoint.y, midPoint.z, a, b);
					saveVariableStruct(a, index, pointersOutA);
					saveVariableStruct(b, index, pointersOutB);
				}
			}
		}


	}

    ///
    /// Loops through each internal (subject to direction) volume cell,
    /// and calls the loop function with additional argument the index of the
    /// neighbouring cells.
    ///
    /// An internal cell is a cell a cell with index 1<j<n"direction", where
    /// j indexes the relevenat direction.
    ///
    /// Direction is given as
    /// direction | description
    /// ----------|------------
    ///     0     | x direction
    ///     1     | y direction
    ///     2     | z direction
    ///
    /// Example
    /// \code{.cpp}
    /// for_each_internal_volume_index<0>(volume, [](size_t l, size_t m, size_t r) {
    ///     // now l is the left index, m is the middle index and r is the right index
    /// });
    /// \endcode
    template<size_t direction>
    inline void for_each_internal_volume_index(const Volume& volume,
                                               const std::function<void(size_t indexLeft, size_t indexMiddle, size_t indexRight)>& function,
											   size_t ghostLayer) {
        static_assert(direction < 3, "We only support up to three dimensions.");
        const bool zDir = direction == 2;
        const bool yDir = direction == 1;
        const bool xDir = direction == 0;

        const size_t nx = volume.getTotalNumberOfXCells();
        const size_t ny = volume.getTotalNumberOfYCells();
        const size_t nz = volume.getTotalNumberOfZCells();

        const size_t startZ = ghostLayer*zDir;
        const size_t startY = ghostLayer*yDir;
        const size_t startX = ghostLayer*xDir;

        const size_t endZ = nz - ghostLayer*zDir;
        const size_t endY = ny - ghostLayer*yDir;
        const size_t endX = nx - ghostLayer*xDir;



        for (size_t z = startZ; z < endZ; z++) {
            for(size_t y = startY; y < endY; y++) {
                for (size_t x = startX; x < endX; x++) {
                    const size_t index = z * nx * ny + y * nx + x;
                    const size_t leftIndex = (z - zDir) * nx * ny
                            + (y - yDir) * nx
                            + (x - xDir);

                    const size_t rightIndex = (z + zDir) * nx * ny
                            + (y + yDir) * nx
                            + (x + xDir);

                    function(leftIndex, index, rightIndex);
                }
            }
        }
    }


	///
	/// Loops through each internal (subject to direction) volume cell,
	/// and calls the loop function with additional argument the index of the
	/// neighbouring cells.
	///
	/// An internal cell is a cell a cell with index 1<j<n"direction", where
	/// j indexes the relevenat direction.
	///
	/// Direction is given as
	/// direction | description
	/// ----------|------------
	///     0     | x direction
	///     1     | y direction
	///     2     | z direction
	///
	/// Example
	/// \code{.cpp}
	/// for_each_internal_volume_index<0>(volume, [](size_t l, size_t m, size_t r) {
	///     // now l is the left index, m is the middle index and r is the right index
	/// });
	/// \endcode
	template<size_t direction>
	inline void for_each_internal_volume_index(const Volume& volume,
		const std::function<void(size_t indexLeft, size_t indexMiddle, size_t indexRight)>& function) {
		for_each_internal_volume_index(volume, function, volume.getNumberOfXGhostCells());
	}
    ///
    /// Works the same way as the templated version, but easier to use in some settings.
    ///
    inline void for_each_internal_volume_index(const Volume& volume, size_t direction,
    const std::function<void(size_t indexLeft, size_t indexMiddle, size_t indexRight)>& function,
	size_t ghostLayer) {

        if (direction == 0) {
            for_each_internal_volume_index<0>(volume, function, ghostLayer);
        } else if (direction == 1) {
			for_each_internal_volume_index<1>(volume, function, ghostLayer);
        } else if (direction == 2) {
			for_each_internal_volume_index<2>(volume, function, ghostLayer);
        } else {
            THROW("We only support direction 0, 1 or 2, was given: " << direction);
        }
    }

	inline void for_each_internal_volume_index(const Volume& volume, size_t direction,
		const std::function<void(size_t indexLeft, size_t indexMiddle, size_t indexRight)>& function) {
		for_each_internal_volume_index(volume, direction,  function, volume.getNumberOfXGhostCells());
	}
}
}
