#pragma once

#include "alsfvm/types.hpp"
#include "alsfvm/boundary/Type.hpp"

namespace alsfvm {
	namespace grid {
		///
		/// Holds the information about the grid
		/// \note We only support regular, uniform cartesian grid
		///
		class Grid {
		public:

			/// 
			/// Constructs the Grid
			/// \param origin the origin point of the grid (the smallest point in lexicographical order)
			/// \param top the top right corner of the grid (maximum point in lexicographical order)
			/// \param dimensions the dimensions of the grid (in number of cells in each direction)
            /// \param boundaryConditions for each side, list the boundary conditions.
            /// Index  |  Spatial side 1D | Spatial side 2D | Spatial side 3D
            /// -------|------------------|-----------------|-----------------
            ///    0   |       left       |     left        |    left
            ///    1   |       right      |     right       |    right
            ///    2   |     < not used > |     bottom      |    bottom
            ///    3   |     < not used > |     top         |    top
            ///    4   |     < not used > |   < not used >  |    front
            ///    5   |     < not used > |   < not used >  |    back
            ///
            Grid(rvec3 origin, rvec3 top, ivec3 dimensions,
                 const std::array<boundary::Type,6>& boundaryConditions = boundary::allPeriodic());

			///
			/// Gets the origin point
			/// \returns the origin point
			///
			rvec3 getOrigin() const;

			///
			/// Gets the top point
			/// \returns the top point
			///
			rvec3 getTop() const;

			///
			/// Gets the dimensions
			/// \returns the dimensions (number of cells in each direction)
			///
			ivec3 getDimensions() const;

            ///
            /// Gets the number of active dimensions (eg. 1D, 2D, 3D)
            /// \returns the number of active space dimensions
            ///
            size_t getActiveDimension() const;

			/// 
			/// Gets the cell lengths in each direction
			///
			rvec3 getCellLengths() const;

			/// 
			/// Returns the midpoints of the grid.
			/// \note this vector is indexed by 
			/// \code{.cpp}
			/// size_t index = z*nx*ny + y*nx + x;
			/// \endcode
			///
			const std::vector<rvec3>& getCellMidpoints() const;


            //! Gets the boundary conditions for the given side
            //!
            //! Index  |  Spatial side 1D | Spatial side 2D | Spatial side 3D
            //! -------|------------------|-----------------|-----------------
            //!    0   |       left       |     left        |    left
            //!    1   |       right      |     right       |    right
            //!    2   |     < not used > |     bottom      |    bottom
            //!    3   |     < not used > |     top         |    top
            //!    4   |     < not used > |   < not used >  |    front
            //!    5   |     < not used > |   < not used >  |    back
            //!
            boundary::Type getBoundaryCondition(int side) const;
		private:
			rvec3 origin;
			rvec3 top;
			ivec3 dimensions;
			rvec3 cellLengths;
			
			// A vector containing all cell midpoints
			std::vector<rvec3> cellMidpoints;

            // For each side, states the boundary condition.
            std::array<boundary::Type,6> boundaryConditions;
		};
	}
}
