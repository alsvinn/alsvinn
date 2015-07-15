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
			virtual bool isOnHost() const;

			///
			/// Gets the pointer to the data (need not be on the host!)
			///
			virtual T* getPointer();

			///
			/// Gets the pointer to the data (need not be on the host!)
			/// \note If this is an OpenCL implementation, the pointer will
			/// be useless! If you want to use the OpenCL memory, you should
			/// first cast to OpenCL memory, then get the OpenCL buffer pointer.
			///
			virtual const T* getPointer() const;

			/// 
			/// Copies the memory to the given buffer
			///
			virtual void copyToHost(T* bufferPointer, size_t bufferLength);


			///
			/// Copies the memory from the buffer (assumed to be on Host/CPU)
			///
			virtual void copyFromHost(const T* bufferPointer, size_t bufferLength);


			///
			/// Adds the other memory area to this one
			/// \param other the memory area to add from
			///
			virtual void operator+=(const Memory<T>& other);

			///
			/// Mutliplies the other memory area to this one
			/// \param other the memory area to multiply from
			///
			virtual void operator*=(const Memory<T>& other);

			///
			/// Subtracts the other memory area to this one
			/// \param other the memory area to subtract from
			///
			virtual void operator-=(const Memory<T>& other);

			///
			/// Divides the other memory area to this one
			/// \param other the memory area to divide from
			///
			virtual void operator/=(const Memory<T>& other);

		private:
			T* memoryPointer;
		};
	}
}