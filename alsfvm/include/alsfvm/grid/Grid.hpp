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

            ///
            Grid(rvec3 origin, rvec3 top, ivec3 dimensions,
                 const std::array<boundary::Type,6>& boundaryConditions = boundary::allPeriodic()
                );

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
            /// \param globalPosition the global position of the current grid in the large grid (used for MPI)
            /// \param globalSize the total size of the grid
            ///
            Grid(rvec3 origin, rvec3 top, ivec3 dimensions,
                 const std::array<boundary::Type,6>& boundaryConditions,
                 const ivec3& globalPosition,
                 const ivec3& globalSize);


            ///
            /// Constructs the Grid
            ///
            /// This is the "least dummy proof version", since it lets the user
            /// specify the cellLengths. This should only be used for domain
            /// decomposition in MPI for instance. Unless you know what you are
            /// doing, don't use this version.
            ///
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
            /// \param globalPosition the global position of the current grid in the large grid (used for MPI)
            /// \param globalSize the total size of the grid
            /// \param cellLengths the cell lengths in each direction
            ///
            /// \note The user is responsible for cellLengths being compatible
            ///       with the rest of the parameters.
            ///
            Grid(rvec3 origin, rvec3 top, ivec3 dimensions,
                 const std::array<boundary::Type,6>& boundaryConditions,
                 const ivec3& globalPosition,
                 const ivec3& globalSize,
                 const rvec3& cellLengths);

            ///
            /// Constructs the Grid
            ///
            /// This is the "least dummy proof version", since it lets the user
            /// specify the cellLengths. This should only be used for domain
            /// decomposition in MPI for instance. Unless you know what you are
            /// doing, don't use this version.
            ///
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
            /// \param globalPosition the global position of the current grid in the large grid (used for MPI)
            /// \param globalSize the total size of the grid
            /// \param cellLengths the cell lengths in each direction
            /// \param cellMidpoints are the cell midpoints with respect to a
            ///                      larger grid, and indexed according to globalPosition
            ///
            /// \note The user is responsible for cellLengths being compatible
            ///       with the rest of the parameters.
            ///
            Grid(rvec3 origin, rvec3 top, ivec3 dimensions,
                 const std::array<boundary::Type,6>& boundaryConditions,
                 const ivec3& globalPosition,
                 const ivec3& globalSize,
                 const rvec3& cellLengths,
                 const std::vector<rvec3>& cellMidpoints);

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

            //! Gets the global position index of the grid,
            //! that is, the grid is contained in a virtual larger grid.
            //! This is used for MPI parallelization to know which part of the
            //! grid we are working on
            ivec3 getGlobalPosition() const;

            //! Get the total size (in number of cells) of the larger grid.
            ivec3 getGlobalSize() const;
		private:
			rvec3 origin;
			rvec3 top;
			ivec3 dimensions;
			rvec3 cellLengths;
			
			// A vector containing all cell midpoints
			std::vector<rvec3> cellMidpoints;

            // For each side, states the boundary condition.
            std::array<boundary::Type,6> boundaryConditions;

            // the global position (in case of using MPI)
            ivec3 globalPosition;

            // the global size
            ivec3 globalSize;
		};
	}
}
