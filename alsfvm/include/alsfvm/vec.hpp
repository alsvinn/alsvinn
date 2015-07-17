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
	};
}