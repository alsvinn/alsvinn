#pragma once
#include "alsfvm/types.hpp"
#include <cassert>

#include <type_traits>
namespace alsfvm {
namespace equation {
namespace euler {

///
/// Holds all the relevant views for the equation.
/// \note We template on VolumeType and ViewType to allow for const and non-const in one.
/// \note We could potentially only template on one of these and use decltype, but there is a
/// bug in MS VC 2013 (http://stackoverflow.com/questions/21609700/error-type-name-is-not-allowed-message-in-editor-but-not-during-compile)
///

template<class VolumeType, class ViewType, int nsd>
class Views {
public:


};

template<class VolumeType, class ViewType>
class Views<VolumeType, ViewType, 3> {
public:
    typedef typename Types<3>::rvec rvec;
    typedef typename std::conditional<std::is_const<VolumeType>::value,
            const real&,
            real&>::type reference_type;

    typedef typename Types<3>::template vec<reference_type> reference_vec;


    Views(VolumeType& volume)
        : rho(volume.getScalarMemoryArea("rho")->getView()),
          mx(volume.getScalarMemoryArea("mx")->getView()),
          my(volume.getScalarMemoryArea("my")->getView()),
          mz(volume.getScalarMemoryArea("mz")->getView()),
          E(volume.getScalarMemoryArea("E")->getView()) {
        // Empty
    }


    template<size_t variableIndex>
    __device__ __host__ ViewType& get() {
        static_assert(variableIndex < 5,
            "We only have 5 conserved variables for Euler!");

        if (variableIndex == 0) {
            return rho;
        } else if (variableIndex == 1) {
            return mx;
        } else if (variableIndex == 2) {
            return my;
        } else if (variableIndex == 3) {
            return mz;

        } else if (variableIndex == 4) {
            return E;
        }

        // If we reach this far, something has gone wrong
        assert(false);
        return rho;
    }


    __device__ __host__ ViewType& get(size_t variableIndex) {
        switch (variableIndex) {
        case 0:
            return rho;

        case 1:
            return mx;

        case 2:
            return my;

        case 3:
            return mz;

        case 4:
            return E;
        }

        // If we reach this far, something has gone wrong
        assert(false);
        return rho;
    }
    __device__ __host__ size_t index(size_t x, size_t y, size_t z) const {
        return rho.index(x, y, z);
    }

    __device__ __host__ reference_vec m(size_t index) {
        return reference_vec(mx.at(index), my.at(index), mz.at(index));
    }

    __device__ __host__ rvec m(size_t index) const {
        return rvec(mx.at(index), my.at(index), mz.at(index));
    }


    ViewType rho;
    ViewType mx;
    ViewType my;
    ViewType mz;
    ViewType E;
};

template<class VolumeType, class ViewType>
class Views<VolumeType, ViewType, 2> {
public:
    typedef typename Types<2>::rvec rvec;
    typedef typename std::conditional<std::is_const<VolumeType>::value,
            const real&,
            real&>::type reference_type;

    typedef typename Types<2>::template vec<reference_type> reference_vec;


    Views(VolumeType& volume)
        : rho(volume.getScalarMemoryArea("rho")->getView()),
          mx(volume.getScalarMemoryArea("mx")->getView()),
          my(volume.getScalarMemoryArea("my")->getView()),
          E(volume.getScalarMemoryArea("E")->getView()) {
        // Empty
    }


    template<size_t variableIndex>
    __device__ __host__ ViewType& get() {
        static_assert(variableIndex < 4,
            "We only have 5 conserved variables for Euler!");

        if (variableIndex == 0) {
            return rho;
        } else if (variableIndex == 1) {
            return mx;
        } else if (variableIndex == 2) {
            return my;
        } else if (variableIndex == 3) {
            return E;
        }

        // If we reach this far, something has gone wrong
        assert(false);
        return rho;
    }


    __device__ __host__ ViewType& get(size_t variableIndex) {
        switch (variableIndex) {
        case 0:
            return rho;

        case 1:
            return mx;

        case 2:
            return my;

        case 3:
            return E;
        }

        // If we reach this far, something has gone wrong
        assert(false);
        return rho;
    }
    __device__ __host__ size_t index(size_t x, size_t y, size_t z) const {
        return rho.index(x, y, z);
    }

    __device__ __host__ reference_vec m(size_t index) {
        return reference_vec(mx.at(index), my.at(index));
    }

    __device__ __host__ rvec m(size_t index) const {
        return rvec(mx.at(index), my.at(index));
    }


    ViewType rho;
    ViewType mx;
    ViewType my;
    ViewType E;
};

template<class VolumeType, class ViewType>
class Views<VolumeType, ViewType, 1> {
public:
    typedef typename Types<1>::rvec rvec;
    typedef typename std::conditional<std::is_const<VolumeType>::value,
            const real&,
            real&>::type reference_type;

    typedef typename Types<1>::template vec<reference_type> reference_vec;


    Views(VolumeType& volume)
        : rho(volume.getScalarMemoryArea("rho")->getView()),
          mx(volume.getScalarMemoryArea("mx")->getView()),
          E(volume.getScalarMemoryArea("E")->getView()) {
        // Empty
    }


    template<size_t variableIndex>
    __device__ __host__ ViewType& get() {
        static_assert(variableIndex < 3,
            "We only have 5 conserved variables for Euler!");

        if (variableIndex == 0) {
            return rho;
        } else if (variableIndex == 1) {
            return mx;
        } else if (variableIndex == 3) {
            return E;
        }

        // If we reach this far, something has gone wrong
        assert(false);
        return rho;
    }


    __device__ __host__ ViewType& get(size_t variableIndex) {
        switch (variableIndex) {
        case 0:
            return rho;

        case 1:
            return mx;

        case 2:
            return E;
        }

        // If we reach this far, something has gone wrong
        assert(false);
        return rho;
    }
    __device__ __host__ size_t index(size_t x, size_t y, size_t z) const {
        return rho.index(x, y, z);
    }

    __device__ __host__ reference_vec m(size_t index) {
        return reference_vec(mx.at(index));
    }

    __device__ __host__ rvec m(size_t index) const {
        return rvec(mx.at(index));
    }


    ViewType rho;
    ViewType mx;
    ViewType E;
};


} // namespace alsfvm
} // namespace equation
} // namespace euler
