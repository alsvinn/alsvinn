#include "alsfvm/grid/Grid.hpp"

namespace alsfvm {
	namespace grid {
		/// 
		/// Constructs the Grid
		/// \param origin the origin point of the grid (the smallest point in lexicographical order)
		/// \param top the top right corner of the grid (maximum point in lexicographical order)
		/// \param dimensions the dimensions of the grid (in number of cells in each direction)
		///
		Grid::Grid(rvec3 origin, rvec3 top, ivec3 dimensions)
			: origin(origin), top(top), dimensions(dimensions)
		{

		}

		///
		/// Gets the origin point
		/// \returns the origin point
		///
		rvec3 Grid::getOrigin() const {
			return origin;
		}

		///
		/// Gets the top point
		/// \returns the top point
		///
		rvec3 Grid::getTop() const {
			return top;
		}

		///
		/// Gets the dimensions
		/// \returns the dimensions (number of cells in each direction)
		///
		ivec3 Grid::getDimensions() const {
			return dimensions;
		}
	}
}