#pragma once
#include "alsfvm/volume/Volume.hpp"
#include <functional>
#include <array>
#include "alsutils/error/Exception.hpp"
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
        /// \param in the volume to loop over
        /// \param function the function to call
        /// \param offsetStart the triple deciding the starting index
        /// \param offsetEnd the offset for end (must be non-negative!)
		///
		template<class Function>
        inline void for_each_cell_index(const Volume& in, const Function& function, ivec3 offsetStart={0,0,0},
                 ivec3 offsetEnd = {0,0,0}) {
            const size_t nx = in.getTotalNumberOfXCells();
            const size_t ny = in.getTotalNumberOfYCells();
            const size_t nz = in.getTotalNumberOfZCells();
            const size_t endx = in.getTotalNumberOfXCells() + offsetEnd[0];
            const size_t endy = in.getTotalNumberOfYCells() + offsetEnd[1];
            const size_t endz = in.getTotalNumberOfZCells() + offsetEnd[2];
            for (size_t k = offsetStart[2]; k < endz; k++) {
                for (size_t j = offsetStart[1]; j < endy; j++) {
                    for (size_t i = offsetStart[0]; i < endx; i++) {
                        size_t index = k*nx*ny + j*nx + i;
						function(index);
					}
				}
			}
		}

        ///
        /// Loops through all possible cell indexes in a cache optimal manner.
        /// \param in the volume to loop over
        /// \param function the function to call
        /// \param offsetStart the triple deciding the starting index
        /// \param offsetEnd the offset for end (must be non-negative!)
        ///
        template<size_t direction, bool parallel=false>
        inline void for_each_cell_index_with_neighbours(const Volume& in,
                                                        const std::function<void(size_t leftIndex, size_t middleIndex, size_t rightIndex)>& function,
                                                        ivec3 offsetStart={0,0,0},
                 ivec3 offsetEnd = {0,0,0}) {
            static_assert(direction < 3, "Direction can be either 0, 1 or 2");
            const bool xDir = direction == 0;
            const bool yDir = direction == 1;
            const bool zDir = direction == 2;

            const auto view = in.getScalarMemoryArea(0)->getView();

            const size_t nx = in.getTotalNumberOfXCells() - offsetEnd[0];
            const size_t ny = in.getTotalNumberOfYCells() - offsetEnd[1];
            const size_t nz = in.getTotalNumberOfZCells() - offsetEnd[2];
            
            for (size_t z = offsetStart[2]; z < nz; z++) {
                for (size_t y = offsetStart[1]; y < ny; y++) {
                    for (size_t x = offsetStart[0]; x < nx; x++) {
                        const size_t index = view.index(x, y, z);
                        const size_t leftIndex = view.index(x - xDir, y - yDir, z - zDir);

                        const size_t rightIndex = view.index(x + xDir, y + yDir, z + zDir);
                        function(leftIndex, index, rightIndex);
                    }
                }
            }
        }


        ///
        /// Loops through all possible cell indexes in a cache optimal manner.
        /// \param in the volume to loop over
        /// \param function the function to call
        /// \param offsetStart the triple deciding the starting index
        /// \param offsetEnd the offset for end (must be non-negative!)
        /// \param direction the direction to use
        /// \note Untemplated version for ease of use, indirectly calls the template version
        ///
        inline void for_each_cell_index_with_neighbours(size_t direction, const Volume& in,
            const std::function<void(size_t leftIndex, size_t middleIndex, size_t rightIndex)>& function,
            ivec3 offsetStart = { 0,0,0 },
            ivec3 offsetEnd = { 0,0,0 }) {
            if (direction == 0) {
                for_each_cell_index_with_neighbours<0>(in, function, offsetStart, offsetEnd);
            }
            else if (direction == 1) {
                for_each_cell_index_with_neighbours<1>(in, function, offsetStart, offsetEnd);
            }
            else if (direction == 2) {
                for_each_cell_index_with_neighbours<2>(in, function, offsetStart, offsetEnd);
            }
            else {
                THROW("Unsupported direction: " << direction);
            }
        }




        template<class VariableStruct>
        inline VariableStruct expandVariableStruct(const std::array<const real*, 0>& in, size_t index) {
            // Yes, we want an empty one, since this is called when we do not
            // have extra variables (eg. burgers)
            return VariableStruct();
        }

        template<class VariableStruct>
        inline VariableStruct expandVariableStruct(const std::array<const real*, 1>& in, size_t index) {
            return VariableStruct(in[0][index]);
        }


        template<class VariableStruct>
        inline VariableStruct expandVariableStruct(const std::array<const real*, 3>& in, size_t index) {
            return VariableStruct(in[0][index], in[1][index], in[2][index]);
        }


        template<class VariableStruct>
        inline VariableStruct expandVariableStruct(const std::array<const real*, 2>& in, size_t index) {
            return VariableStruct(in[0][index], in[1][index]);
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
        inline void saveVariableStruct(const VariableStruct& in, size_t index, std::array<real*, 1>& out) {
            real* inAsRealPointer = (real*)&in;

            out[0][index] = inAsRealPointer[0];

        }

        template<class VariableStruct>
        inline void saveVariableStruct(const VariableStruct& in, size_t index, std::array<real*, 0>& out) {
            // Yes, we want an empty one, since this is called when we do not
            // have extra variables (eg. burgers)
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
        inline void saveVariableStruct(const VariableStruct& in, size_t index, std::array<real*, 3>& out) {
            real* inAsRealPointer = (real*)&in;

            out[0][index] = inAsRealPointer[0];
            out[1][index] = inAsRealPointer[1];
            out[2][index] = inAsRealPointer[2];

        }

        template<class VariableStruct>
        inline void saveVariableStruct(const VariableStruct& in, size_t index, std::array<real*, 2>& out) {
            real* inAsRealPointer = (real*)&in;

            out[0][index] = inAsRealPointer[0];
            out[1][index] = inAsRealPointer[1];

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

        const size_t nx = outA.getTotalNumberOfXCells();
        const size_t ny = outA.getTotalNumberOfYCells();
        const size_t nz = outA.getTotalNumberOfZCells();

        const size_t nxGrid = grid.getDimensions().x;
        const size_t nyGrid = grid.getDimensions().y;

		auto& midPoints = grid.getCellMidpoints();

        const size_t ghostX = outA.getNumberOfXGhostCells();
        const size_t ghostY = outA.getNumberOfYGhostCells();
        const size_t ghostZ = outA.getNumberOfZGhostCells();
        for (size_t k = ghostZ; k < nz - ghostZ; k++) {
            for (size_t j = ghostY; j < ny - ghostY; j++) {
                for (size_t i = ghostX; i < nx - ghostX; i++) {
                    size_t index = k*nx*ny + j*nx + i;
                    size_t midpointIndex = (k - ghostZ) * nxGrid * nyGrid
                            + (j - ghostY) * nxGrid + (i - ghostX);
                    auto midPoint = midPoints[midpointIndex];
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
    /// Loops through each cell midpoint, and call function
    /// with the coordinates and the corresponding index in the volume.
    ///
    inline void for_each_midpoint(const Volume& volume,
                                  const grid::Grid& grid,
                                  const std::function < void(real x, real y, real z, size_t index, size_t i, size_t j, size_t k) > & function) {

        const size_t nx = volume.getTotalNumberOfXCells();
        const size_t ny = volume.getTotalNumberOfYCells();
        const size_t nz = volume.getTotalNumberOfZCells();

        const size_t nxGrid = grid.getDimensions().x;
        const size_t nyGrid = grid.getDimensions().y;

        auto& midPoints = grid.getCellMidpoints();

        const size_t ghostX = volume.getNumberOfXGhostCells();
        const size_t ghostY = volume.getNumberOfYGhostCells();
        const size_t ghostZ = volume.getNumberOfZGhostCells();

        for (size_t k = ghostZ; k < nz - ghostZ; k++) {
            for (size_t j = ghostY; j < ny - ghostY; j++) {
                for (size_t i = ghostX; i < nx - ghostX; i++) {
                    size_t index = k*nx*ny + j*nx + i;

                    size_t midpointIndex = (k - ghostZ) * nxGrid * nyGrid
                            + (j - ghostY) * nxGrid + (i - ghostX);
                    auto midPoint = midPoints[midpointIndex];

                    function(midPoint.x, midPoint.y, midPoint.z, index, i-ghostX, j-ghostY, k-ghostZ);
                }
            }
        }


    }

    ///
    /// Loops through each cell midpoint, and call function
    /// with the coordinates and the corresponding index in the volume.
    ///
    /// \note This just passes through to the other function taking a bigger lambda signature.
    ///
    inline void for_each_midpoint(const Volume& volume,
                                  const grid::Grid& grid,
                                  const std::function < void(real x, real y, real z, size_t index) > & function) {
        for_each_midpoint(volume, grid, [&](real x, real y, real z, size_t index, size_t, size_t, size_t) {
            function(x,y,z,index);
        });
    }


    ///
    /// Fill the volume based on a filler function (depending on position).
    /// Example (making a simple Riemann problem in 2D):
    /// \code{.cpp}
    /// fill_volume<ConservedVariables>(conserved, grid,
    ///    [](real x, real y, real z, ConservedVariables& outConserved) {
    ///        if ( x < 0.5) {
    ///           outConserved.rho = 1.0;
    ///           outConserved.m = rvec3(2,1,1);
    ///           outConserved.E = 8;
    ///        } else {
    ///           outConserved.rho = 1.2;
    ///           outConserved.m = rvec3(2,1,1);
    ///           outConserved.E = 9;
    ///        }
    ///    });
    /// \endcode
    ///
template<class VariableStruct>
inline void fill_volume(Volume& out,
    const grid::Grid& grid,
    const std::function < void(real x, real y, real z, VariableStruct& out) > & fillerFunction) {

    std::array<real*, sizeof(VariableStruct) / sizeof(real)> pointersOut;

    if (pointersOut.size() != out.getNumberOfVariables()) {
        THROW("We expected to get " << pointersOut.size() << " variables, but got " << out.getNumberOfVariables());
    }



    for (size_t i = 0; i < out.getNumberOfVariables(); i++) {
        pointersOut[i] = out.getScalarMemoryArea(i)->getPointer();
    }

    for_each_midpoint(out, grid, [&](real x, real y, real z, size_t index) {
        VariableStruct a;
        fillerFunction(x, y, z, a);
        saveVariableStruct(a, index, pointersOut);
    });


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
    /// and calls the loop function
    ///
    /// Example
    /// \code{.cpp}
    /// for_each_internal_volume_indexvolume, [](size_t l, size_t m, size_t r) {
    ///     // now l is the left index, m is the middle index and r is the right index
    /// });
    /// \endcode
    inline void for_each_internal_volume_index(const Volume& volume,
                                               const std::function<void( size_t indexMiddle)>& function) {


        const size_t ngx = volume.getNumberOfXGhostCells();
        const size_t ngy = volume.getNumberOfYGhostCells();
        const size_t ngz = volume.getNumberOfZGhostCells();
        const size_t nx = volume.getTotalNumberOfXCells();
        const size_t ny = volume.getTotalNumberOfYCells();
        const size_t nz = volume.getTotalNumberOfZCells();

        const size_t startZ = ngz;
        const size_t startY = ngy;
        const size_t startX = ngx;

        const size_t endZ = nz - ngz;
        const size_t endY = ny - ngy;
        const size_t endX = nx - ngx;



        for (size_t z = startZ; z < endZ; z++) {
            for(size_t y = startY; y < endY; y++) {
                for (size_t x = startX; x < endX; x++) {
                    const size_t index = z * nx * ny + y * nx + x;


                    function(index);
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
        for_each_internal_volume_index<direction>(volume, function, volume.getNumberOfXGhostCells());
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
