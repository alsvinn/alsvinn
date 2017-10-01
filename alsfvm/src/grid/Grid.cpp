#include "alsfvm/grid/Grid.hpp"
#include <algorithm>
#include <iostream>
namespace alsfvm {
	namespace grid {

    namespace {
        rvec3 computeCellLengths(const rvec3& origin, const rvec3& top,
                                 const ivec3 dimensions) {
            int dimensionX = dimensions[0];
            int dimensionY = dimensions[1];
            int dimensionZ = dimensions[2];

            auto lengths = (top-origin) / rvec3(dimensionX, dimensionY, dimensionZ);

            if (dimensionY == 1) {
                lengths[1] = 1;
            }

            if (dimensionZ == 1) {
                lengths[2] = 1;
            }

            return lengths;
        }
    }
		/// 
		/// Constructs the Grid
		/// \param origin the origin point of the grid (the smallest point in lexicographical order)
		/// \param top the top right corner of the grid (maximum point in lexicographical order)
		/// \param dimensions the dimensions of the grid (in number of cells in each direction)
		///
        Grid::Grid(rvec3 origin, rvec3 top, ivec3 dimensions,
                   const std::array<boundary::Type, 6>& boundaryConditions)

			: origin(origin), top(top), dimensions(dimensions),
            cellLengths(computeCellLengths(origin, top, dimensions)),
            boundaryConditions(boundaryConditions)
		{
			// Create the cell midpoints
			cellMidpoints.resize(dimensions.x*dimensions.y*dimensions.z);
			
            for (int z = 0; z < dimensions.z; z++) {
                for (int y = 0; y < dimensions.y; y++) {
                    for (int x = 0; x < dimensions.x; x++) {
                        rvec3 position = origin
                                + rvec3(cellLengths.x*x, cellLengths.y*y, cellLengths.z*z)
                                + cellLengths / real(2.0);

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

        boundary::Type Grid::getBoundaryCondition(int side) const
        {
            return boundaryConditions[side];
        }
	}
}
