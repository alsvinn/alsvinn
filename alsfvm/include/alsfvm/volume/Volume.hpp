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
			/// \param numberOfGhostCells the number of ghost cells
			///
			/// \note we deduce from ny and nz whether or not to added ghostcells
			///       in that direction. Ie. if ny==1, then we do not add ghost cells in y direction
            ///
            Volume(const std::vector<std::string>& variableNames,
                   alsfvm::shared_ptr<memory::MemoryFactory> memoryFactory,
                   size_t nx, size_t ny, size_t nz,
				   size_t numberOfGhostCells = 0);

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
            alsfvm::shared_ptr<memory::Memory<real> >&
                getScalarMemoryArea(size_t index);

            ///
            /// \brief getScalarMemoryArea gets the scalar memory area (real)
            /// \param index the index of the variable. Use getIndexFromName
            ///              to get the index.
            ///
            /// \return the MemoryArea for the given index
            ///
            alsfvm::shared_ptr<const memory::Memory<real> >
                getScalarMemoryArea(size_t index) const;

			///
			/// \brief getScalarMemoryArea gets the scalar memory area (real)
			/// \param name the name of the variable
			/// \return the MemoryArea for the given name
            /// \note Equivalent to calling
            ///     getScalarMemoryArea(getIndexFromName(name))
			///
			alsfvm::shared_ptr<memory::Memory<real> >&
				getScalarMemoryArea(const std::string& name);

            ///
            /// \brief getScalarMemoryArea gets the scalar memory area (real)
            /// \param name the name of the variable
            /// \return the MemoryArea for the given name
            /// \note Equivalent to calling
            ///     getScalarMemoryArea(getIndexFromName(name))
            ///
            alsfvm::shared_ptr<const memory::Memory<real> >
                getScalarMemoryArea(const std::string& name) const;


            ///
            /// \brief getIndexFromName returns the given index from the name
            /// \param name the name of the variable
            /// \return the index of the name.
            ///
            size_t getIndexFromName(const std::string& name) const ;

			///
			/// Gets the variable name associated to the given index
			/// \param index the index of the variable name
			/// \returns the variable name
			/// \note This implicitly uses the std::move-feature of C++11
			///
            std::string getName(size_t index) const;

			///
			/// Adds each component of the other volume to this volume
			///
			Volume& operator+=(const Volume& other);


			/// 
			/// Multiplies each component of the volume by the scalar
			///
			Volume& operator*=(real scalar);

			///
			/// \returns the number of cells in X direction
			///
			size_t getNumberOfXCells() const;

			///
			/// \returns the number of cells in Y direction
			///
			size_t getNumberOfYCells() const;

			///
			/// \returns the number of cells in Z direction
			///
			size_t getNumberOfZCells() const;

            ///
            /// \brief makeZero sets every element of the volume to zero (0).
            ///
            void makeZero();

			///
			/// Gets the number of ghost cells in x direction
			/// \note This is the number of ghost cells on one side.
			///
			size_t getNumberOfXGhostCells() const;

			///
			/// Gets the number of ghost cells in y direction
			/// \note This is the number of ghost cells on one side.
			///
			size_t getNumberOfYGhostCells() const;

			///
			/// Gets the number of ghost cells in z direction
			/// \note This is the number of ghost cells on one side.
			///
			size_t getNumberOfZGhostCells() const;

			///
			/// Returns the total number of cells in x direction, including ghost cells
			///
			size_t getTotalNumberOfXCells() const;

			///
			/// Returns the total number of cells in y direction, including ghost cells
			///
			size_t getTotalNumberOfYCells() const;

			///
			/// Returns the total number of cells in z direction, including ghost cells
			///
			size_t getTotalNumberOfZCells() const;

            ///
            /// Copies the contents of the given memory area into the buffer output.
            ///
            /// This is ideal for removing the ghost cells before output.
            ///
            /// \note Throws an exception if outputSize < number of cells
            ///
            void copyInternalCells(size_t memoryAreaIndex, real* output, size_t outputSize) const;

			/// 
			/// Copies the whole volume to the other volume
			///
			void copyTo(volume::Volume& other) const;


        private:
            const std::vector<std::string> variableNames;
            const alsfvm::shared_ptr<memory::MemoryFactory> memoryFactory;

            std::vector<alsfvm::shared_ptr<memory::Memory<real> > >
                memoryAreas;
			size_t nx;
			size_t ny;
			size_t nz;

			
			size_t numberOfXGhostCells;
			size_t numberOfYGhostCells;
			size_t numberOfZGhostCells;
		};
	}
}
