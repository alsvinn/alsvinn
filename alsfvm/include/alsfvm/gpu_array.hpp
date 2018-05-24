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

///
/// \brief The gpu_array class is akin to the std::array, only also works for
///        gpus
///
template<class T, size_t N>
class gpu_array {
public:

    __host__ __device__ gpu_array() {
        // empty
    }

    __host__ __device__ gpu_array(std::initializer_list<T> initializerList) {
        int i = 0;

        for (const T& t : initializerList) {
            data[i++] = t;
        }
    }

    __host__ __device__ T& operator[](int i) {
        return data[i];
    }

    __host__ __device__ const T& operator[](int i) const {
        return data[i];
    }

    size_t size() const {
        return N;
    }
private:
    T data[N];

};



}
