#include "alsfvm/boundary/NeumannCPU.hpp"
#include "alsfvm/error/Exception.hpp"
#include <cassert>

namespace alsfvm { namespace boundary { 
	namespace {
		/// 
		/// small helper function
		/// \param volume the volume to apply neumann boundary conditions to
		/// \param dimensions the number of dimensions(1,2 or 3).
		///
		void applyNeumann(volume::Volume& volume, const size_t dimensions, const size_t numberOfGhostCells) {
            const int nx = volume.getTotalNumberOfXCells();
            const int ny = volume.getTotalNumberOfYCells();
            const int nz = volume.getTotalNumberOfZCells();

			if (nx < 2 * numberOfGhostCells || ((dimensions > 1) && ny < 2 * numberOfGhostCells) || ((dimensions > 2) && nz < 2 * numberOfGhostCells) ) {
				THROW("Too few cells to apply boundary condition. We got (nx, ny, nz)=("
					<< nx << ", " << ny << ", " << nz << "), but numberOfGhostCells = " << numberOfGhostCells
					<< std::endl << std::endl << "We require at least double the number of cells in each direction");
			}

			// We need these two assertions for this to work (else we need fancier indexing)
			assert(nx*sizeof(real) == volume.getScalarMemoryArea(0)->getExtentXInBytes());
			assert(ny*sizeof(real) == volume.getScalarMemoryArea(0)->getExtentYInBytes());

			// loop through variables
			for (int var = 0; var < volume.getNumberOfVariables(); var++) {
				real* pointer = volume.getScalarMemoryArea(var)->getPointer();
				// loop through dimensions
				for (int d = 0; d < dimensions; d++) {
					// i=0 represents bottom, i=1 represents top
					for (int i = 0; i < 2; i++) {
						const bool zDir = d == 2;
						const bool yDir = d == 1;
						const bool xDir = d == 0;
                        // Either we start on the left (i == 0), or on the right(i==1)
                        const size_t zStart = zDir ?
                            (i == 0 ? numberOfGhostCells : nz - numberOfGhostCells - 1) : 0;

                        const size_t zEnd = zDir ?
                            (zStart + 1) : nz;

                        const size_t yStart = yDir ?
                            (i == 0 ? numberOfGhostCells : ny - numberOfGhostCells - 1) : 0;

                        const size_t yEnd = yDir ?
                            (yStart + 1) : ny;

                        const size_t xStart = xDir ?
                            (i == 0 ? numberOfGhostCells : nx - numberOfGhostCells - 1) : 0;

                        const size_t xEnd = xDir ?
                            (xStart + 1) : nx;

                        for (int z = zStart; z < zEnd; z++) {
                            for (int y = yStart; y < yEnd; y++) {
                                for (int x = xStart; x < xEnd; x++) {
                                    for (size_t ghostCell = 0; ghostCell < numberOfGhostCells; ghostCell++) {
                                        const size_t sourceIndex =
                                              (z + zDir * ((1-2*i)*ghostCell)) * nx * ny
                                            + (y + yDir * ((1-2*i)*ghostCell)) * nx
                                            + (x + xDir * ((1-2*i)*ghostCell));

                                        const size_t targetIndex =
                                              (z + (2* i - 1) * zDir * (ghostCell + 1)) * nx * ny
                                            + (y + (2* i - 1) * yDir * (ghostCell + 1)) * nx
                                            + (x + (2* i - 1) * xDir * (ghostCell + 1));
                                        pointer[targetIndex] = pointer[sourceIndex];
                                    }
                                }
                            }
                        }


					}
				}
			}
		}
	}

	NeumannCPU::NeumannCPU(size_t numberOfGhostCells)
		: numberOfGhostCells(numberOfGhostCells) 
	{
		// empty
	}

	void NeumannCPU::applyBoundaryConditions(volume::Volume& volume, const grid::Grid& grid)
	{
		applyNeumann(volume, grid.getActiveDimension(), numberOfGhostCells);

	}
}
}
