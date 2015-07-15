#pragma once

/// 
/// Various vector operations in CUDA
///

namespace alsfvm {
	namespace cuda {
		
		///
		/// Adds a and b and stores the result to result
		/// \param result the device memory to write to (can be the same
		///               as a or b). Must have length size (in T)
		/// \param a must have length size (in T)
		/// \param b must have length size (in T)
		///
		template<class T>
		void add(T* result, const T* a, const T* b, size_t size);

		///
		/// Multiplies a and b and stores the result to result
		/// \param result the device memory to write to (can be the same
		///               as a or b). Must have length size (in T)
		/// \param a must have length size (in T)
		/// \param b must have length size (in T)
		///
		template<class T>
		void multiply(T* result, const T* a, const T* b, size_t size);

		///
		/// Subtracts a from b and stores the result to result
		/// \param result the device memory to write to (can be the same
		///               as a or b). Must have length size (in T)
		/// \param a must have length size (in T)
		/// \param b must have length size (in T)
		///
		template<class T>
		void subtract(T* result, const T* a, const T* b, size_t size);

		///
		/// Divides a by b and stores the result to result
		/// \param result the device memory to write to (can be the same
		///               as a or b). Must have length size (in T)
		/// \param a must have length size (in T)
		/// \param b must have length size (in T)
		///
		template<class T>
		void divide(T* result, const T* a, const T* b, size_t size);
	}
}