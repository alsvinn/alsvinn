#pragma once
#include "alsfvm/types.hpp"

namespace alsfvm {
namespace equation {
namespace cubic {
///
/// Holds all the relevant views for the equation.
/// \note We template on VolumeType and ViewType to allow for const and non-const in one.
/// \note We could potentially only template on one of these and use decltype, but there is a
/// bug in MS VC 2013 (http://stackoverflow.com/questions/21609700/error-type-name-is-not-allowed-message-in-editor-but-not-during-compile)
///
template<class VolumeType, class ViewType>
class Views {
public:

    Views(VolumeType& volume)
        : u(volume.getScalarMemoryArea("u")->getView()) {
        // Empty
    }


    template<size_t variableIndex>
    __device__ __host__ ViewType& get() {
        static_assert(variableIndex < 1,
            "We only have 1 conserved variables for Cubic!");
        return u;
    }


    __device__ __host__ ViewType& get(size_t variableIndex) {
        assert(variableIndex == 0);
        return u;
    }
    __device__ __host__ size_t index(size_t x, size_t y, size_t z) const {
        return u.index(x, y, z);
    }


    ViewType u;
};

} // namespace alsfvm
} // namespace equation
} // namespace cubic
