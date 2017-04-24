#pragma once

namespace alsfvm { namespace equation { namespace cubic { 

///
/// Holds all the relevant views for the equation (extra variables)
/// \note We template on VolumeType and ViewType to allow for const and non-const in one.
/// \note We could potentially only template on one of these and use decltype, but there is a
/// bug in MS VC 2013 (http://stackoverflow.com/questions/21609700/error-type-name-is-not-allowed-message-in-editor-but-not-during-compile)
///
template<class VolumeType, class ViewType>
class ViewsExtra {
public:

    ViewsExtra(VolumeType& volume)
    {
        // Empty
    }
};

} // namespace alsfvm
} // namespace equation
} // namespace cubic
