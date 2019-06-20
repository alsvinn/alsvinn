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
#include "alsfvm/volume/Volume.hpp"
#include <vector>

namespace alsfvm {
namespace volume {

//! Easy reference to the combination of conserved volume and extra volume
class VolumePair {
public:
    VolumePair() {}
    typedef std::vector<std::shared_ptr<volume::Volume>>::iterator IteratorType;
    typedef std::vector<std::shared_ptr<volume::Volume>>::const_iterator
        ConstIteratorType;

    VolumePair(std::shared_ptr<volume::Volume> conservedVolume,
        std::shared_ptr<volume::Volume> extraVolume);


    VolumePair(std::shared_ptr<volume::Volume> conservedVolume);

    std::shared_ptr<volume::Volume> getConservedVolume();
    std::shared_ptr<volume::Volume> getExtraVolume();

    IteratorType begin();
    IteratorType end();


    ConstIteratorType begin() const;
    ConstIteratorType end() const;
private:
    std::vector<std::shared_ptr<volume::Volume>> volumes;

};
} // namespace volume
} // namespace alsfvm
