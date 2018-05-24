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

#include "alsfvm/volume/VolumePair.hpp"

namespace alsfvm {
namespace volume {

VolumePair::VolumePair(std::shared_ptr<volume::Volume> conservedVolume,
    std::shared_ptr<volume::Volume> extraVolume)
    : volumes{{conservedVolume, extraVolume}} {

}

std::shared_ptr<Volume> VolumePair::getConservedVolume() {
    return volumes[0];
}

std::shared_ptr<Volume> VolumePair::getExtraVolume() {
    return volumes[1];
}

VolumePair::IteratorType VolumePair::begin() {
    return volumes.begin();
}

VolumePair::IteratorType VolumePair::end() {
    return volumes.end();
}

VolumePair::ConstIteratorType VolumePair::begin() const {
    return volumes.begin();
}

VolumePair::ConstIteratorType VolumePair::end() const {
    return volumes.end();
}

}
}
