#pragma once
#include <cstdlib>
#include <cctype>
#include <cstdlib>
#include <memory>
#include "alsfvm/vec.hpp"
#ifdef ALSVINN_USE_QUADMATH
#include <quadmath.h>
#endif
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

    ///
    /// Computes the square of x
    /// \returns x*x
    ///
    inline real square(const real& x) {
        return x * x;
    }
}

#ifdef ALSVINN_USE_QUADMATH
namespace std {
    inline __float128 abs(const __float128& x) {
        return fabsq(x);
    }

    inline bool isnan(const __float128& x) {
        return isnanq(x);
    }

    inline bool pow(__float128 x, int b) {
        return powq(x, b);
    }

    inline bool pow(__float128 x, double b) {
        return powq(x, b);
    }

    inline bool isfinite(const __float128& x) {
        return !isinfq(x);
    }

    inline bool isinf(const __float128& x) {
        return isinfq(x);
    }

    inline std::basic_ostream<char>& operator<<(std::basic_ostream<char>& os, const __float128& x) {
        return (os << (double)x);
    }


}
#endif
