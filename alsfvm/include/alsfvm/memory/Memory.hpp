#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/memory/MemoryBase.hpp"
namespace alsfvm {
	namespace memory {
		///
		/// Class to hold data. Do note that this is an abstract interface,
		/// look at the other concrete implementations to use this. 
		///
        template<class T>
		class Memory : public MemoryBase {
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

            ///
            /// @returns the size (in bytes) of the memory in X direction.
            /// \note use this for indexing
            /// <code>
            /// const size_t extentX = memory.getExtentXInBytes();
            /// const size_t extentY = memory.getExtentYInBytes();
            /// size_t indexByte = i*extentX*ExtentY+j*extentX+k;
            /// </code>
            ///
            virtual size_t getExtentXInBytes() const;

            ///
            /// @returns the size (in bytes) of the memory in Y direction.
            /// \note use this for indexing
            /// <code>
            /// const size_t extentX = memory.getExtentXInBytes();
            /// const size_t extentY = memory.getExtentYInBytes();
            /// size_t indexByte = i*extentX*ExtentY+j*extentX+k;
            /// </code>
            ///
            virtual size_t getExtentYInBytes() const;

			///
			/// Checks if the memory area is on the host (CPU) or 
			/// on some device, if the latter, one needs to copy to host
			/// before reading it.
			/// @returns true if the memory is on host, false otherwise
			///
			virtual bool isOnHost() const = 0;

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
			
		protected:
            const size_t nx;
            const size_t ny;
            const size_t nz;
		};
	}
}
