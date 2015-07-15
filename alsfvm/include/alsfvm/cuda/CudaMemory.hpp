#include "alsfvm/memory/Memory.hpp"
namespace alsfvm {
	namespace cuda {

		template<class T>
		class CudaMemory : public memory::Memory<T> {
		public:
			CudaMemory(size_t size);

			// Note: Virtual distructor since we will inherit
			// from this. 
			virtual ~CudaMemory();

			///
			/// Checks if the memory area is on the host (CPU) or 
			/// on some device, if the latter, one needs to copy to host
			/// before reading it.
			/// @returns false
			///
			virtual bool isOnHost();

			///
			/// Gets the pointer to the data (need not be on the host!)
			///
			virtual T* getPointer();

			/// 
			/// Copies the memory to the given buffer
			///
			virtual void copyToHost(T* bufferPointer, size_t bufferLength);


			///
			/// Copies the memory from the buffer (assumed to be on Host/CPU)
			///
			virtual void copyFromHost(const T* bufferPointer, size_t bufferLength);
		private:
			T* memoryPointer;
		};
	}
}