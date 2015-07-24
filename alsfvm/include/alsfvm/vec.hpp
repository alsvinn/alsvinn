#pragma once
#include <vector>
namespace alsfvm {

	///
	/// Small vector class to hold 3D data
	///
	template<class T>
	struct vec3 {
		T x;
		T y;
		T z;

		vec3(T x, T y, T z) 
		: x(x), y(y), z(z)
		{
			// Empty
		}

		bool operator==(const vec3& other) const {
			return other.x == x && other.y == y && other.z == z;
		}

		///
		/// Converts the vector to an std::vector<T>
		/// output is {x, y, z}
		///
		std::vector<T> toStdVector() {
			return std::vector<T>({ x, y, z });
		}

		/// 
		/// Converts the vector to the other type
		///
		template<class S>
		vec3<S> convert() {
			return vec3<S>(S(x), S(y), S(z));
		}

		///
		/// Returns the i-th component of the vector.
		///
		const T& operator[](size_t i) const {
			// Note: We only store three numbers in this struct, hence this is safe
			return ((T*)this)[i];
		}

		///
		/// Returns the i-th component of the vector.
		///
		T& operator[](size_t i) {
			// Note: We only store three numbers in this struct, hence this is safe
			return ((T*)this)[i];
		}

        ///
        /// Computes the dot (scalar) product
        ///
         dot(const vec3<T>& other) const {
            return x*other.x + y*other.y + z*other.z;
        }
	};

	///
	/// Computes the product \f$\mathrm{scalar} \vec{a}\f$
	/// \note Creates a new vector instance
	///
	template<class T>
	inline vec3<T> operator*(T scalar, const vec3<T>& a) {
		return vec3<T>(a.x*scalar, a.y*scalar, a.z*scalar);
	}

	///
	/// Computes the division \f$\frac{1}{\mathrm{scalar}} \vec{a}\f$
	/// \note Creates a new vector instance
	///
	template<class T>
	inline vec3<T> operator/(const vec3<T>& a,T scalar) {
		return vec3<T>(a.x/scalar, a.y/scalar, a.z/scalar);
	}

	///
	/// Computes the product \f$\vec{a}+\vec{b}\f$
	/// \note Creates a new vector instance.
	///
	template<class T>
	inline vec3<T> operator+(const vec3<T>& a, const vec3<T>& b) {
		return vec3<T>(a.x+b.x, a.y + b.y,  a.z + b.z);
	}
}
