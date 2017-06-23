#pragma once
#include <memory>
#include "alsfvm/types.hpp"
#include "alsfvm/memory/MemoryBase.hpp"
#include "alsfvm/memory/View.hpp"
namespace alsfvm {
	namespace memory {
		///
		/// Class to hold data. Do note that this is an abstract interface,
		/// look at the other concrete implementations to use this. 
		///
        template<class T>
        class Memory : public MemoryBase, public std::enable_shared_from_this<Memory<T> > {
		public:


            ///
            /// \brief Memory constructs new memory
            /// \param nx the number of cells in x direction
            /// \param ny the number of cells in y direction
            /// \param nz the number of cells in z direction
            ///
            Memory(size_t nx, size_t ny, size_t nz);

			// Note: Virtual distructor since we will inherit
			// from this. 
			virtual ~Memory() {}

			///
            /// @returns the size (in number of T) of the memory
			///
            virtual size_t getSize() const;

            ///
            /// @returns the size (in number of T) of the memory in X diretion
            ///
            virtual size_t getSizeX() const;

            ///
            /// @returns the size (in number of T) of the memory in Y diretion
            ///
            virtual size_t getSizeY() const;

            ///
            /// @returns the size (in number of T) of the memory in Z diretion
            ///
            virtual size_t getSizeZ() const;

            //! Clones the memory area, but *does not copy the content*
            virtual std::shared_ptr<Memory<T> > makeInstance() const = 0;

            //! Copies the contents of the other memory area into this one
            virtual void copyFrom(const Memory<T>& other) = 0;

            ///
            /// @returns the size (in bytes) of the memory in X direction.
            /// \note use this for indexing
            /// \code{.cpp}
            /// const size_t extentX = memory.getExtentXInBytes();
            /// const size_t extentY = memory.getExtentYInBytes();
            /// size_t indexByte = i*extentX*ExtentY+j*extentX+k;
            /// \endcode
            ///
            virtual size_t getExtentXInBytes() const;

            ///
            /// @returns the size (in bytes) of the memory in Y direction.
            /// \note use this for indexing
            /// \code{.cpp}
            /// const size_t extentX = memory.getExtentXInBytes();
            /// const size_t extentY = memory.getExtentYInBytes();
            /// size_t indexByte = i*extentX*ExtentY+j*extentX+k;
            /// \endcode
            ///
            virtual size_t getExtentYInBytes() const;

			///
			/// Checks if the memory area is on the host (CPU) or 
			/// on some device, if the latter, one needs to copy to host
			/// before reading it.
			/// @returns true if the memory is on host, false otherwise
			///
			virtual bool isOnHost() const = 0;

            T* data() {
                return getPointer();
            }

            const T* data() const {
                return getPointer();
            }

            T& operator[](size_t i) {
                assert(isOnHost());
                return data()[i];
            }

            T operator[](size_t i) const {
                return data()[i];
            }

			///
			/// Gets the pointer to the data (need not be on the host!)
            /// \note If this is an OpenCL implementation, the pointer will
            /// be useless! If you want to use the OpenCL memory, you should
            /// first cast to OpenCL memory, then get the OpenCL buffer pointer.
			///
            virtual T* getPointer() = 0;

			///
			/// Gets the pointer to the data (need not be on the host!)
			/// \note If this is an OpenCL implementation, the pointer will
			/// be useless! If you want to use the OpenCL memory, you should
			/// first cast to OpenCL memory, then get the OpenCL buffer pointer.
			///
			virtual const T* getPointer() const = 0;

			/// 
			/// Copies the memory to the given buffer
            /// \note bufferLength must be at least getSize()
            /// \param bufferPointer the buffer to write to
            /// \param bufferLength the size of the buffer (in number of T's)
			///
            virtual void copyToHost(T* bufferPointer,
                                    size_t bufferLength) const = 0;

			///
			/// Copies the memory from the buffer (assumed to be on Host/CPU)
            /// \note bufferLength must be at least getSize()
            /// \param bufferPointer the buffer to write to
            /// \param bufferLength the size of the buffer (in number of T's)
            ///
            virtual void copyFromHost(const T* bufferPointer,
                                      size_t bufferLength) = 0;

			///
			/// Adds the other memory area to this one
			/// \param other the memory area to add from
			///
			virtual void operator+=(const Memory<T>& other) = 0;

			///
			/// Mutliplies the other memory area to this one
			/// \param other the memory area to multiply from
			///
			virtual void operator*=(const Memory<T>& other) = 0;

			///
			/// Subtracts the other memory area to this one
			/// \param other the memory area to subtract from
			///
			virtual void operator-=(const Memory<T>& other) = 0;

			///
			/// Divides the other memory area to this one
			/// \param other the memory area to divide from
			///
			virtual void operator/=(const Memory<T>& other) = 0;


			///
			/// Adds the scalar to each component
			/// \param scalar the scalar to add
			///
			virtual void operator+=(real scalar) = 0;

			///
			/// Multiplies the scalar to each component
			/// \param scalar the scalar to multiply
			///
			virtual void operator*=(real scalar) = 0;

			///
			/// Subtracts the scalar from each component
			/// \param scalar the scalar to subtract
			///
			virtual void operator-=(real scalar) = 0;

			///
			/// Divides the each component by the scalar
			/// \param scalar the scalar to divide
			///
			virtual void operator/=(real scalar) = 0;

            ///
            /// \brief makeZero sets every element to zero (0)
            ///
            virtual void makeZero() = 0;

            ///
            /// \brief copyInternalCells copies the internal cells into the memory area
            /// This is ideal for removing ghost cells before outputing the solution.
            /// \param startX start index (inclusive) for x direction
            /// \param endX end index (exclusive) for x direction
            /// \param startY start index (inclusive) for y direction
            /// \param endY end index (exclusive) for y direction
            /// \param startZ start index (inclusive) for z direction
            /// \param endZ end index (exclusive) for z direction
            /// \param output the output buffer
            /// \param outputSize must be at least the size of the written memory
            ///
            /// This is essentially equivalent to doing
            /// \code{.cpp}
            /// size_t numberOfZ = endZ-startZ;
            /// size_t numberOfY = endY-startY;
            /// size_t numberOfX = endX-startX;
            ///
            /// for(size_t z = startZ; z < endZ; z++) {
            ///     for(size_t y = startY; y < endY; y++) {
            ///         for(size_t x = startX; x < endX; x++) {
            ///
            ///             size_t indexIn = z * nx * ny + y * nx + x;
            ///             size_t indexOut = (z-startZ) * numberOfX * numberOfY
            ///                   + (y - startY) * numberOfY + (x - startX);
            ///             output[indexOut] = data[indexIn];
            ///          }
            ///     }
            /// }
            /// \endcode
            ///
            virtual void copyInternalCells(size_t startX, size_t endX,
                                           size_t startY, size_t endY,
                                           size_t startZ, size_t endZ,
                                           T* output, size_t outputSize) = 0;

            ///
            /// \brief getView gets the view to the memory
            /// \return the view
            ///
            View<T> getView();


            ///
            /// \brief getView gets the view to the memory (const version)
            /// \return the view
            ///
            View<const T> getView() const;


            //! Adds the memory with coefficients to this memory area
            //! Here we compute the sum
            //! \f[ v_1^{\mathrm{new}}=a_1v_1+a_2v_2+a_3v_3+a_4v_4+a_5v_5+a_6v_6\f]
            //! where \f$v_1\f$ is the volume being operated on.
            virtual void addLinearCombination(T a1, 
                T a2, const Memory<T>& v2, 
                T a3, const Memory<T>& v3,
                T a4, const Memory<T>& v4, 
                T a5, const Memory<T>& v5) = 0;



            //! Adds a power of the other memory area to this memory area, ie
            //!
            //! \f[this += pow(other, power)\f]
            //!
            //! @param other the other memory area to the the power of
            //! @param power the power to use
            virtual void addPower(const Memory<T>& other, double power) = 0;


            //! Subtracts a power of the other memory area to this memory area, ie
            //!
            //! \f[this -= pow(other, power)\f]
            //!
            //! @param other the other memory area to the the power of
            //! @param power the power to use
            virtual void subtractPower(const Memory<T>& other, double power) = 0;


            //! Copies the data to host if it is on GPU, otherwise makes a copy
            virtual std::shared_ptr<Memory<T> > getHostMemory() = 0;

            //! Computes the total variation, given here as
            //!
            //! \f[\sum_{i,j,k} \sqrt(\sum_{n=1}^d|u_{(i,j,k)}-u_{(i,j,k)-e_n}|^2)^p.\f]
            //!
            //! \note This function gives no performance guarantees
            //!
            //! @param p the exponent p
            //! @param start the index to start at (inclusive)
            //! @param end the maximum index (exclusive)
            virtual real getTotalVariation(int p, const ivec3& start,
                                           const ivec3& end) const = 0;

            //! Computes the total variation in a given direction \$d\in\{0,1,2\}\$
            //!
            //! \f[\sum_{i,j,k} |u_{(i,j,k)}-u_{(i,j,k)-e_n}|^p.\f]
            //!
            //! \note This function gives no performance guarantees
            //!
            //! @param p the exponent p
            //! @param direction the direction (between 0 and 2 inclusive)
            //! @param start the index to start at (inclusive)
            //! @param end the maximum index (exclusive)
            virtual real getTotalVariation(int direction, int p, const ivec3& start,
                                           const ivec3& end) const = 0;


			
		protected:
            const size_t nx;
            const size_t ny;
            const size_t nz;
		};
	}
}
