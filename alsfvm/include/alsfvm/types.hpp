#pragma once
#include <cctype>
#include <cstdlib>
#include <memory>
#include "alsfvm/vec.hpp"
namespace alsfvm {
	typedef double real;
	typedef vec3<real> rvec3;
	typedef vec3<int> ivec3;

	///
	/// The available types we have
	///
	enum Types {
		REAL
	};
}
