#include "alsfvm/volume/Volume.hpp"
#include "alsfvm/error/Exception.hpp"

namespace alsfvm {
	namespace volume {

		Volume::Volume(const std::vector<std::string>& variableNames,
			boost::shared_ptr<memory::MemoryFactory> memoryFactory,
			size_t nx, size_t ny, size_t nz,
			size_t numberOfGhostCells)
			: variableNames(variableNames), memoryFactory(memoryFactory), nx(nx), ny(ny), nz(nz),
			numberOfXGhostCells(numberOfGhostCells), 
			numberOfYGhostCells(ny > 1 ? numberOfGhostCells : 0),
			numberOfZGhostCells(nz > 1 ? numberOfGhostCells : 0)
		{
            for (size_t i=0; i < variableNames.size(); i++) {
                memoryAreas.push_back(memoryFactory->createScalarMemory(
					nx + 2 * numberOfXGhostCells, 
					ny + 2 * numberOfYGhostCells,
					nz + 2 * numberOfZGhostCells));
            }
		}


		Volume::~Volume()
		{
            // Everything is deleted automatically
		}

		size_t Volume::getNumberOfVariables() const {
			return variableNames.size();
		}

		///
		/// \brief getScalarMemoryArea gets the scalar memory area (real)
		/// \param index the index of the variable. Use getIndexFromName
		///              to get the index.
		///
		/// \return the MemoryArea for the given index
		///
		boost::shared_ptr<memory::Memory<real> >&
			Volume::getScalarMemoryArea(size_t index) {
			return memoryAreas[index];
		}

        ///
        /// \brief getScalarMemoryArea gets the scalar memory area (real)
        /// \param index the index of the variable. Use getIndexFromName
        ///              to get the index.
        ///
        /// \return the MemoryArea for the given index
        ///
        boost::shared_ptr<const memory::Memory<real> >
            Volume::getScalarMemoryArea(size_t index) const {
            return memoryAreas[index];
        }

		///
		/// \brief getScalarMemoryArea gets the scalar memory area (real)
		/// \param name the name of the variable
		/// \return the MemoryArea for the given name
		/// \note Equivalent to calling getScalarMemoryArea(getIndexFromName(name))
		///
		boost::shared_ptr<memory::Memory<real> >&
			Volume::getScalarMemoryArea(const std::string& name) {
			return getScalarMemoryArea(getIndexFromName(name));
		}

        ///
        /// \brief getScalarMemoryArea gets the scalar memory area (real)
        /// \param name the name of the variable
        /// \return the MemoryArea for the given name
        /// \note Equivalent to calling getScalarMemoryArea(getIndexFromName(name))
        ///
        boost::shared_ptr<const memory::Memory<real> >
            Volume::getScalarMemoryArea(const std::string& name) const {
            return getScalarMemoryArea(getIndexFromName(name));
        }

		///
		/// \brief getIndexFromName returns the given index from the name
		/// \param name the name of the variable
		/// \return the index of the name.
		///
        size_t Volume::getIndexFromName(const std::string& name) const {

			// Do this simple for now
			for (size_t i = 0; i < variableNames.size(); i++) {
				if (variableNames[i] == name) {
					return i;
				}
			}
			THROW("Could not find variable name: " << name);
		}


		///
		/// Gets the variable name associated to the given index
		/// \param index the index of the variable name
		/// \returns the variable name
		/// \note This implicitly uses the std::move-feature of C++11
		///
        std::string Volume::getName(size_t index) const {
			return variableNames[index];
		}


		///
		/// Adds each component of the other volume to this volume
		///
		Volume& Volume::operator+=(const Volume& other) {
			for (size_t i = 0; i < memoryAreas.size(); i++) {
				(*(memoryAreas)[i]) += *(other.getScalarMemoryArea(i));
			}
			return *this;
		}


		/// 
		/// Multiplies each component of the volume by the scalar
		///
		Volume& Volume::operator*=(real scalar) {
			for (size_t i = 0; i < memoryAreas.size(); i++) {
				(*(memoryAreas)[i]) *= scalar;
			}
			return *this;
		}

		///
		/// \returns the number of cells in X direction
		///
		size_t Volume::getNumberOfXCells() const {
			return nx;
		}

		///
		/// \returns the number of cells in Y direction
		///
		size_t Volume::getNumberOfYCells() const {
			return ny;
		}

		///
		/// \returns the number of cells in Z direction
		///
		size_t Volume::getNumberOfZCells() const {
            return nz;
        }

        void Volume::copyInternalCells(size_t memoryAreaIndex, real *output, size_t outputSize) const
        {
            memoryAreas[memoryAreaIndex]->copyInternalCells(numberOfXGhostCells, getTotalNumberOfXCells() - numberOfXGhostCells,
                                           numberOfYGhostCells, getTotalNumberOfYCells() - numberOfYGhostCells,
                                           numberOfZGhostCells, getTotalNumberOfZCells() - numberOfZGhostCells, output, outputSize);
        }

        void Volume::makeZero()
        {
            for(size_t i = 0; i < memoryAreas.size(); i++) {
                memoryAreas[i]->makeZero();
            }
        }

		///
		/// Gets the number of ghost cells in x direction
		/// \note This is the number of ghost cells on one side.
		///
		size_t Volume::getNumberOfXGhostCells() const {
			return numberOfXGhostCells;
		}

		///
		/// Gets the number of ghost cells in y direction
		/// \note This is the number of ghost cells on one side.
		///
		size_t Volume::getNumberOfYGhostCells() const {
			return numberOfYGhostCells;
		}

		///
		/// Gets the number of ghost cells in z direction
		/// \note This is the number of ghost cells on one side.
		///
		size_t Volume::getNumberOfZGhostCells() const {
			return numberOfZGhostCells;
		}

		///
		/// Returns the total number of cells in x direction, including ghost cells
		///
		size_t Volume::getTotalNumberOfXCells() const {
			return nx + 2 * numberOfXGhostCells;
		}

		///
		/// Returns the total number of cells in y direction, including ghost cells
		///
		size_t Volume::getTotalNumberOfYCells() const {
			return ny + 2 * numberOfYGhostCells;
		}

		///
		/// Returns the total number of cells in z direction, including ghost cells
		///
		size_t Volume::getTotalNumberOfZCells() const {
			return nz + 2 * numberOfZGhostCells;
		}


		/// 
		/// Copies the whole volume to the other volume
		///
		void Volume::copyTo(volume::Volume& other) const {
			std::vector<real> temporaryStorage(getScalarMemoryArea(0)->getSize());
			for (size_t var = 0; var < getNumberOfVariables(); ++var) {
				getScalarMemoryArea(var)->copyToHost(temporaryStorage.data(), temporaryStorage.size());
				other.getScalarMemoryArea(var)->copyFromHost(temporaryStorage.data(), temporaryStorage.size());
			}
		}
	}

}
