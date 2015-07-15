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
            /// @param size the size of the memory area (in number of T)
			///
			Memory(size_t size);

			// Note: Virtual distructor since we will inherit
			// from this. 
			virtual ~Memory() {}

			///
            /// @returns the size (in number of T) of the memory
			///
			size_t getSize() const;

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
                                    size_t bufferLength) = 0;

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
			
		protected:
			size_t size;
		};
	}
}
