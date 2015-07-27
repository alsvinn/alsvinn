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
			: origin(origin), top(top), dimensions(dimensions),
			cellLengths((top - origin) / dimensions.convert<real>())
		{
			
			// Create the cell midpoints
			cellMidpoints.resize(dimensions.x*dimensions.y*dimensions.z);
			
			for (size_t z = 0; z < dimensions.z; z++) {
				for (size_t y = 0; y < dimensions.y; y++) {
					for (size_t x = 0; x < dimensions.x; x++) {
						rvec3 position = origin + rvec3(cellLengths.x*x, cellLengths.y*y, cellLengths.z*z) + cellLengths / 2.0;
						cellMidpoints[z*dimensions.x*dimensions.y + y*dimensions.x + x] = position;
					}
				}
			}
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

        size_t Grid::getActiveDimension() const
        {
            if (dimensions.z > 1) {
                return 3;
            } else if (dimensions.y > 1) {
                return 2;
            } else {
                return 1;
            }
        }

		/// 
		/// Gets the cell lengths in each direction
		///
		rvec3 Grid::getCellLengths() const {
			return cellLengths;
		}

		const std::vector<rvec3>& Grid::getCellMidpoints() const {
			return cellMidpoints;
		}
	}
}
