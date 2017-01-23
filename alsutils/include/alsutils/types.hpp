#pragma once
#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <cstdlib>
#include <cfloat>
#include <memory>
#include <complex>
#ifndef _WIN32
// On Linux, we can not use C++11, so we use boost
// on Windows, the compiler doesn't like boost that much at the moment.
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
namespace alsfvm {
	using boost::shared_ptr;
	using boost::make_shared;
	using boost::dynamic_pointer_cast;
}
#else
#include <memory>
namespace alsfvm {
	using std::shared_ptr;
	using std::make_shared;
	using std::dynamic_pointer_cast;
}
#endif
// For CUDA we need special flags for the functions, 
// for normal build, we just need to define these flags as empty.
#ifdef ALSVINN_HAVE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>

#include "alsfvm/gpu_array.hpp"
#else 

#define __device__ 
#define __host__
#endif

#if __cplusplus <= 199711L
#ifndef _WIN32
#include <cassert>
#define static_assert(x, y) assert(x)
#endif
#endif

#include "alsutils/vec.hpp"
#include "alsutils/mat.hpp"
#ifdef ALSVINN_USE_QUADMATH
#include <quadmath.h>
#endif


namespace alsutils {

    typedef double real;


    typedef vec1<real> rvec1;
    typedef vec1<int> ivec1;

    typedef vec2<real> rvec2;
    typedef vec2<int> ivec2;

	typedef vec3<real> rvec3;
	typedef vec3<int> ivec3;

    typedef vec4<real> rvec4;
    typedef vec4<int> ivec4;

    typedef vec5<real> rvec5;
    typedef vec5<int> ivec5;
    
    typedef vec6<real> rvec6;
    typedef vec6<int> ivec6;

    inline __device__ __host__ ivec3 make_direction_vector(size_t direction) {
        return ivec3( direction == 0, direction == 1, direction == 2 );
    }

    typedef matrix<real, 1, 1> matrix1;

    typedef matrix<real, 2, 2> matrix2;
    typedef matrix<real, 3, 3> matrix3;
    typedef matrix<real, 4, 4> matrix4;
    typedef matrix<real, 5, 5> matrix5;

	template<int nsd>
	struct Types {
		// empty
	};

    template<>
    struct Types<1> {
        typedef rvec1 rvec;
        typedef ivec1 ivec;
        template<class T>
        using vec = vec1<T>;
        typedef matrix1 matrix;
    };

    template<>
    struct Types<2> {
        typedef rvec2 rvec;
        typedef ivec2 ivec;
        template<class T>
        using vec = vec2<T>;
        typedef matrix2 matrix;
    };

    template<>
    struct Types<3> {
        typedef rvec3 rvec;
        typedef ivec3 ivec;
        template<class T>
        using vec = vec3<T>;
        typedef matrix3 matrix;
    };

    template<>
    struct Types<4> {
        typedef rvec4 rvec;
        typedef ivec4 ivec;
        template<class T>
        using vec = vec4<T>;
        typedef matrix4 matrix;
    };

    template<>
    struct Types<5> {
        typedef rvec5 rvec;
        typedef ivec5 ivec;
        template<class T>
        using vec = vec5<T>;
        typedef matrix5 matrix;
    };



    ///
    /// Computes the square of x
    /// \returns x*x
    ///
    inline __host__ __device__ real square(const real& x) {
        return x * x;
    }


}

#ifdef ALSVINN_USE_QUADMATH
namespace std {

    inline __float128 fabs(const __float128& x) {
        return fabsq(x);
    }

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
