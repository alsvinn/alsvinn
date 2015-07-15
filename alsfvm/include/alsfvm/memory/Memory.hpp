#pragma once
#include "alsfvm/types.hpp"
namespace alsfvm {
	namespace memory {
		///
		/// Class to hold data. Do note that this is an abstract interface,
		/// look at the other concrete implementations to use this. 
		///
		class Memory {
		public:
			///
			/// @param size the size of the memory area (in bytes)
			///
			Memory(size_t size);

			// Note: Virtual distructor since we will inherit
			// from this. 
			virtual ~Memory() {}

			///
			/// @returns the size (in bytes) of the memory
			///
			size_t getSize();

			///
			/// Checks if the memory area is on the host (CPU) or 
			/// on some device, if the latter, one needs to copy to host
			/// before reading it.
			/// @returns true if the memory is on host, false otherwise
			///
			virtual bool isOnHost() = 0;

			///
			/// Gets the pointer to the data (need not be on the host!)
			///
			virtual void* getPointer() = 0; 

			/// 
			/// Copies the memory to the given buffer
			///
			virtual void copyToHost(void* bufferPointer, size_t bufferLength) = 0;

			///
			/// Copies the memory from the buffer (assumed to be on Host/CPU)
			///
			virtual void copyFromHost(const void* bufferPointer, size_t bufferLength) = 0;
			
		protected:
			size_t size;
		};
	}
}