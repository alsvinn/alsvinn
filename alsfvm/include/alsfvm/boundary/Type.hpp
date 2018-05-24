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
#include <array>

namespace alsfvm {
namespace boundary {

enum Type {
    //! Periodic boundary conditions
    PERIODIC,

    //! Neumann boundary
    NEUMANN,

    //! Boundary is handled by the MPI system (gotten from another process)
    MPI_BC
};

//! Convenience function to create an array of length 6 with only Type::PERIODIC
//! entries
inline std::array<Type, 6> allPeriodic() {
    return {{
            Type::PERIODIC, Type::PERIODIC, Type::PERIODIC,
            Type::PERIODIC, Type::PERIODIC, Type::PERIODIC
        }};
}


//! Convenience function to create an array of length 6 with only Type::NEUMANN
//! entries
inline std::array<Type, 6> allNeumann() {
    return {{
            Type::NEUMANN, Type::NEUMANN, Type::NEUMANN,
            Type::NEUMANN, Type::NEUMANN, Type::NEUMANN
        }};
}

} // namespace boundary
} // namespace alsfvm
