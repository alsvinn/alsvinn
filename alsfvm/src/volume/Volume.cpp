#include "alsfvm/volume/Volume.hpp"
#include "alsfvm/error/Exception.hpp"

namespace alsfvm {
	namespace volume {

		Volume::Volume(const std::vector<std::string>& variableNames,
			std::shared_ptr<memory::MemoryFactory> memoryFactory,
			size_t nx, size_t ny, size_t nz)
			: variableNames(variableNames), memoryFactory(memoryFactory)
		{
            for (size_t i=0; i < variableNames.size(); i++) {
                memoryAreas.push_back(memoryFactory->createScalarMemory(nx, ny, nz));
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
		std::shared_ptr<memory::Memory<real> >&
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
        std::shared_ptr<const memory::Memory<real> >
            Volume::getScalarMemoryArea(size_t index) const {
            return memoryAreas[index];
        }

		///
		/// \brief getScalarMemoryArea gets the scalar memory area (real)
		/// \param name the name of the variable
		/// \return the MemoryArea for the given name
		/// \note Equivalent to calling getScalarMemoryArea(getIndexFromName(name))
		///
		std::shared_ptr<memory::Memory<real> >&
			Volume::getScalarMemoryArea(const std::string& name) {
			return getScalarMemoryArea(getIndexFromName(name));
		}

        ///
        /// \brief getScalarMemoryArea gets the scalar memory area (real)
        /// \param name the name of the variable
        /// \return the MemoryArea for the given name
        /// \note Equivalent to calling getScalarMemoryArea(getIndexFromName(name))
        ///
        std::shared_ptr<const memory::Memory<real> >
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
	}

}
