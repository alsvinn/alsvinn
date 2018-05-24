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

#include "alsfvm/reconstruction/ENOCoefficients.hpp"

namespace alsfvm {
namespace reconstruction {

template<>
real ENOCoeffiecients<1>::coefficients[][1] = {
    {1},
    {1}
};

template<>
real ENOCoeffiecients<2>::coefficients[][2] = {
    {3.0 / 2.0, -1.0 / 2.0},
    {1.0 / 2.0,  1.0 / 2.0},
    {-1.0 / 2.0, 3.0 / 2.0}
};

template<>
real ENOCoeffiecients<3>::coefficients[][3] = {
    {11.0 / 6.0, -7.0 / 6.0, 1.0 / 3.0},
    {1.0 / 3.0, 5.0 / 6.0, -1.0 / 6.0},
    {-1.0 / 6.0, 5.0 / 6.0, 1.0 / 3.0},
    {1.0 / 3.0, -7.0 / 6.0, 11.0 / 6.0}
};

template<>
real ENOCoeffiecients<4>::coefficients[][4] = {
    {25.0 / 12.0,  -23.0 / 12.0,  13.0 / 12.0, -1.0 / 4.0},
    { 1.0 / 4.0,    13.0 / 12.0,  -5.0 / 12.0,  1.0 / 12.0},
    {-1.0 / 12.0,    7.0 / 12.0,   7.0 / 12.0, -1.0 / 12.0},
    { 1.0 / 12.0,   -5.0 / 12.0,  13.0 / 12.0,  1.0 / 4.0},
    {-1.0 / 4.0,    13.0 / 12.0, -23.0 / 12.0, 25.0 / 12.0}
};

}
}

