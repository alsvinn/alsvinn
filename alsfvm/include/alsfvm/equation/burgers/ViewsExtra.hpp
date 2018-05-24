/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

namespace alsfvm {
namespace equation {
namespace burgers {

///
/// Holds all the relevant views for the equation (extra variables)
/// \note We template on VolumeType and ViewType to allow for const and non-const in one.
/// \note We could potentially only template on one of these and use decltype, but there is a
/// bug in MS VC 2013 (http://stackoverflow.com/questions/21609700/error-type-name-is-not-allowed-message-in-editor-but-not-during-compile)
///
template<class VolumeType, class ViewType>
class ViewsExtra {
public:

    ViewsExtra(VolumeType& volume) {
        // Empty
    }
};

} // namespace alsfvm
} // namespace equation
} // namespace burgers
