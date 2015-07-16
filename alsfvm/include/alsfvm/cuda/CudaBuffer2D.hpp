#pragma once
#include "alsfvm/cuda/CudaBuffer.hpp"
namespace alsfvm {
	namespace cuda {

		///
		/// 1D cuda buffer
		///
		template<typename T>
		class CudaBuffer2D : public CudaBuffer < T > {
		public:
			///
			/// \param nx the number of Ts in x direction
			/// \param ny the number of Ts in y direction
			///
			CudaBuffer2D(size_t nx, size_t ny);
			virtual ~CudaBuffer2D();


			/// 
			/// \returns the pointer to the data
			///
			virtual T* getPointer();

			/// 
			/// \returns the pointer to the data
			///
			virtual const T* getPointer() const;


		private:
			T* memoryPointer;
		};
	}
}