#pragma once
#include "alsfvm/volume/Volume.hpp"
namespace alsfvm { namespace volume { 

///
/// Holds all the primitive variables for an Euler simulation.
///
/// The extra variables are
/// \f[V = \left(\begin{array}{c} p\\u_x\\u_y\\u_z\\ \rho\end{array}\right)\f]
/// where \f$p\f$ is the pressure, \f$u_x\f$, \f$u_y\f$ and \f$u_z\f$
/// is the velocity in
/// \f$x, y, z\f$-direction, and \f$\rho\f$ is the density.
///
class EulerPrimitiveVolume : public Volume {
public:
    ///
    /// Typedef to make some function signatures look nicer,
    /// nothing else.
    ///
    typedef std::shared_ptr<memory::Memory<real> > ScalarMemoryPtr;

    ///
    /// Const version of the memory pointer
    ///
    typedef std::shared_ptr<const memory::Memory<real> > ConstScalarMemoryPtr;

    ///
    /// Constructs the EulerVolume
    ///
    /// \param memoryFactory the memory factory to use when creating new memory areas
    /// \param nx the number of cells in x direction
    /// \param ny the number of cells in y direction
    /// \param nz the number of cells in z direction
    /// \param numberOfGhostCells the number of ghostcells to use
    ///
    EulerPrimitiveVolume(std::shared_ptr<memory::MemoryFactory> memoryFactory,
        size_t nx, size_t ny, size_t nz, size_t numberOfGhostCells);



};
} // namespace alsfvm
} // namespace volume
