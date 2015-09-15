#include "alsfvm/memory/Memory.hpp"
namespace alsfvm {
	namespace cuda {

		template<class T>
		class CudaMemory : public memory::Memory<T> {
		public:
			///
			/// \param nx the number of cells in x direction
			/// \param ny the number of cells in y direction
			/// \param nz the number of cells in z direction
			///
			CudaMemory(size_t nx, size_t ny=1, size_t nz=1);

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
			virtual void copyToHost(T* bufferPointer, size_t bufferLength) const;


			///
			/// Copies the memory from the buffer (assumed to be on Host/CPU)
			///
			virtual void copyFromHost(const T* bufferPointer, size_t bufferLength);


			///
			/// Adds the other memory area to this one
			/// \param other the memory area to add from
			///
		        virtual void operator+=(const memory::Memory<T>& other);

			///
			/// Mutliplies the other memory area to this one
			/// \param other the memory area to multiply from
			///
        		virtual void operator*=(const memory::Memory<T>& other);

			///
			/// Subtracts the other memory area to this one
			/// \param other the memory area to subtract from
			///
                        virtual void operator-=(const memory::Memory<T>& other);

			///
			/// Divides the other memory area to this one
			/// \param other the memory area to divide from
			///
		        virtual void operator/=(const memory::Memory<T>& other);


			///
			/// Adds the scalar to each component
			/// \param scalar the scalar to add
			///
			virtual void operator+=(real scalar);

			///
			/// Multiplies the scalar to each component
			/// \param scalar the scalar to multiply
			///
			virtual void operator*=(real scalar);

			///
			/// Subtracts the scalar from each component
			/// \param scalar the scalar to subtract
			///
			virtual void operator-=(real scalar);

			///
			/// Divides the each component by the scalar
			/// \param scalar the scalar to divide
			///
			virtual void operator/=(real scalar);

			///
			/// Sets every component to zero
			///
			virtual void makeZero();


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
			/// This calls cudaMemcpy3d behind the scenes.
			///
			virtual void copyInternalCells(size_t startX, size_t endX,
				size_t startY, size_t endY,
				size_t startZ, size_t endZ,
				T* output, size_t outputSize);
		private:
			T* memoryPointer;
		};
	}
}
