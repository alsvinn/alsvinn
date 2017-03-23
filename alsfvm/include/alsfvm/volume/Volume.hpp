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

            //! Make a volume as a view of another volume.
            //! @param volume the volume to make a view of
            //! @param components the components to use
            //! @param variableNames the variableNames to use
            Volume(Volume& volume, const std::vector<size_t>& components,
                const std::vector<std::string>& variableNames);

            // We need a virtual destructor in case we want to inherit from this
			virtual ~Volume();

            ///
            /// \brief getNumberOfVariables gets the number of variables used
            /// \return the number of variables
            ///
            virtual size_t getNumberOfVariables() const;

            ///
            /// \brief getScalarMemoryArea gets the scalar memory area (real)
            /// \param index the index of the variable. Use getIndexFromName
            ///              to get the index.
            ///
            /// \return the MemoryArea for the given index
            ///
            virtual alsfvm::shared_ptr<memory::Memory<real> >&
                getScalarMemoryArea(size_t index);

            ///
            /// \brief getScalarMemoryArea gets the scalar memory area (real)
            /// \param index the index of the variable. Use getIndexFromName
            ///              to get the index.
            ///
            /// \return the MemoryArea for the given index
            ///
            virtual alsfvm::shared_ptr<const memory::Memory<real> >
                getScalarMemoryArea(size_t index) const;

			///
			/// \brief getScalarMemoryArea gets the scalar memory area (real)
			/// \param name the name of the variable
			/// \return the MemoryArea for the given name
            /// \note Equivalent to calling
            ///     getScalarMemoryArea(getIndexFromName(name))
			///
            virtual alsfvm::shared_ptr<memory::Memory<real> >&
				getScalarMemoryArea(const std::string& name);

            ///
            /// \brief getScalarMemoryArea gets the scalar memory area (real)
            /// \param name the name of the variable
            /// \return the MemoryArea for the given name
            /// \note Equivalent to calling
            ///     getScalarMemoryArea(getIndexFromName(name))
            ///
            virtual alsfvm::shared_ptr<const memory::Memory<real> >
                getScalarMemoryArea(const std::string& name) const;

            ///
            /// \brief getScalarMemoryArea gets the scalar memory area (real)
            /// \param index the index of the variable
            /// \return the MemoryArea for the given name
            /// \note Equivalent to calling
            ///     getScalarMemoryArea(index)
            ///
            virtual alsfvm::shared_ptr<const memory::Memory<real> >
                operator[](size_t index) const;

            ///
            /// \brief getScalarMemoryArea gets the scalar memory area (real)
            /// \param index the index of the variable
            /// \return the MemoryArea for the given name
            /// \note Equivalent to calling
            ///     getScalarMemoryArea(index)
            ///
            virtual alsfvm::shared_ptr<memory::Memory<real> >
                operator[](size_t index);



            ///
            /// \brief getIndexFromName returns the given index from the name
            /// \param name the name of the variable
            /// \return the index of the name.
            ///
            virtual size_t getIndexFromName(const std::string& name) const ;

			///
			/// Gets the variable name associated to the given index
			/// \param index the index of the variable name
			/// \returns the variable name
			/// \note This implicitly uses the std::move-feature of C++11
			///
            virtual std::string getName(size_t index) const;

			///
			/// Adds each component of the other volume to this volume
			///
            virtual Volume& operator+=(const Volume& other);


			/// 
			/// Multiplies each component of the volume by the scalar
			///
            virtual Volume& operator*=(real scalar);

			///
			/// \returns the number of cells in X direction
			///
            virtual size_t getNumberOfXCells() const;

			///
			/// \returns the number of cells in Y direction
			///
            virtual size_t getNumberOfYCells() const;

			///
			/// \returns the number of cells in Z direction
			///
            virtual size_t getNumberOfZCells() const;

            ///
            /// \brief makeZero sets every element of the volume to zero (0).
            ///
            virtual void makeZero();

			///
			/// Gets the number of ghost cells in x direction
			/// \note This is the number of ghost cells on one side.
			///
            virtual size_t getNumberOfXGhostCells() const;

			///
			/// Gets the number of ghost cells in y direction
			/// \note This is the number of ghost cells on one side.
			///
            virtual size_t getNumberOfYGhostCells() const;

			///
			/// Gets the number of ghost cells in z direction
			/// \note This is the number of ghost cells on one side.
			///
            virtual size_t getNumberOfZGhostCells() const;

			///
			/// Returns the total number of cells in x direction, including ghost cells
			///
            virtual size_t getTotalNumberOfXCells() const;

			///
			/// Returns the total number of cells in y direction, including ghost cells
			///
            virtual size_t getTotalNumberOfYCells() const;

			///
			/// Returns the total number of cells in z direction, including ghost cells
			///
            virtual size_t getTotalNumberOfZCells() const;

            ///
            /// Copies the contents of the given memory area into the buffer output.
            ///
            /// This is ideal for removing the ghost cells before output.
            ///
            /// \note Throws an exception if outputSize < number of cells
            ///
            virtual void copyInternalCells(size_t memoryAreaIndex, real* output, size_t outputSize) const;

			/// 
			/// Copies the whole volume to the other volume
			///
            virtual void copyTo(volume::Volume& other) const;

            ///
            /// \brief setVolume sets the contents of this volume to the contenst of the other volume
            /// \param other the other volume to read from
            /// \note This does interpolation if necessary.
            ///
            virtual void setVolume(const volume::Volume& other);

            //! Gets the number of space dimensions.
            virtual size_t getDimensions() const;


            //! Adds the volumes with coefficients to this volume
            //! Here we compute the sum
            //! \f[ v_1^{\mathrm{new}}=a_1v_1+a_2v_2+a_3v_3+a_4v_4+a_5v_5+a_6v_6\f]
            //! where \f$v_1\f$ is the volume being operated on.
            void addLinearCombination(real a1, real a2, const Volume& v2, real a3, const Volume& v3, real a4, const Volume& v4, real a5, const Volume& v5);


            //! Gets the total size in each dimension.
            //! Equivalent to calling
            //! \code{.cpp}
            //! ivec3 dimensions{volume.getTotalNumberOfXCells(),
            //!                  volume.getTotalNumberOfYCells(),
            //!                  volume.getTotalNumberOfZCells()};
            //! \endcode
            virtual ivec3 getTotalDimensions() const;


            //! Adds a power of the other volume to this volume, ie
            //!
            //! \f[this += pow(other, power)\f]
            //!
            //! @param other the other volume to the the power of
            //! @param power the power to use
            virtual void addPower(const Volume& other, real power);

            //! Subtracts a power of the other volume to this volume, ie
            //!
            //! \f[this -= pow(other, power)\f]
            //!
            //! @param other the other volume to the the power of
            //! @param power the power to use
            virtual void subtractPower(const Volume& other, real power);


            //! Makes a volume with the same memory areas and the same sizes
            std::shared_ptr<volume::Volume> makeInstance() const;

            //! Makes a new volume with the same names for the memory areas,
            //! but with the newly given sizes.
            std::shared_ptr<volume::Volume> makeInstance(size_t nx, size_t ny, size_t nz, const std::string& platform = "default") const;


        private:
            const std::vector<std::string> variableNames;
            alsfvm::shared_ptr<memory::MemoryFactory> memoryFactory;
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
