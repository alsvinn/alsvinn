#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/memory/MemoryFactory.hpp"
#include "alsfvm/memory/Memory.hpp"
#include <string>
#include <vector>

namespace alsfvm {
	namespace volume {
        ///
        /// \brief The Volume class represents a volume (a collection of cells
        /// with values for each cell (eg. pressure, density, etc)
        ///
        ///
		class Volume
		{
		public:
            ///
            /// \brief Volume creates a new volume object
            /// \param variableNames a list of the used variable names.
            ///                      Typically this can be "rho", "m", "u",
            ///                      etc.
            ///
            /// \param memoryFactory the memory factory to use to create new
            ///                      memory areas.
            ///
            /// \param nx the number of cells in x diretion
            /// \param ny the number of cells in y diretion
            /// \param nz the number of cells in z diretion
            ///
            Volume(const std::vector<std::string>& variableNames,
                   std::shared_ptr<memory::MemoryFactory> memoryFactory,
                   size_t nx, size_t ny, size_t nz);

            // We need a virtual destructor in case we want to inherit from this
			virtual ~Volume();

            ///
            /// \brief getNumberOfVariables gets the number of variables used
            /// \return the number of variables
            ///
			size_t getNumberOfVariables() const;

            ///
            /// \brief getScalarMemoryArea gets the scalar memory area (real)
            /// \param index the index of the variable. Use getIndexFromName
            ///              to get the index.
            ///
            /// \return the MemoryArea for the given index
            ///
            std::shared_ptr<memory::Memory<real> >&
                getScalarMemoryArea(size_t index);

			///
			/// \brief getScalarMemoryArea gets the scalar memory area (real)
			/// \param name the name of the variable
			/// \return the MemoryArea for the given name
			/// \note Equivalent to calling getScalarMemoryArea(getIndexFromName(name))
			///
			std::shared_ptr<memory::Memory<real> >&
				getScalarMemoryArea(const std::string& name);

            ///
            /// \brief getIndexFromName returns the given index from the name
            /// \param name the name of the variable
            /// \return the index of the name.
            ///
            size_t getIndexFromName(const std::string& name);

			///
			/// Gets the variable name associated to the given index
			/// \param index the index of the variable name
			/// \returns the variable name
			/// \note This implicitly uses the std::move-feature of C++11
			///
			std::string getName(size_t index);


		private:
            const std::shared_ptr<memory::MemoryFactory> memoryFactory;
            const std::vector<std::string> variableNames;
            std::vector<std::shared_ptr<memory::Memory<real> > >
                memoryAreas;
		};
	}
}
