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
