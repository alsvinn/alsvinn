#pragma once
#include "alsfvm/volume/Volume.hpp"
namespace alsfvm { namespace volume { 

	///
	/// Holds all the extra variables for an Euler simulation.
	///
	/// The extra variables are
	/// \f[V = \left(\begin{array}{c} p\\u_x\\u_y\\u_z\end{array}\right)\f]
	/// where \f$p\f$ is the pressure, \f$u_x\f$, \f$u_y\f$ and \f$u_z\f$ 
	/// is the velocity in 
	/// \f$x, y, z\f$-direction.
	/// 
    class EulerExtraVolume : public Volume {
    public:
		///
		/// Typedef to make some function signatures look nicer,
		/// nothing else.
		///
		typedef alsfvm::shared_ptr<memory::Memory<real> > ScalarMemoryPtr;

		///
		/// Const version of the memory pointer
		///
		typedef alsfvm::shared_ptr<const memory::Memory<real> > ConstScalarMemoryPtr;

		///
		/// Constructs the EulerVolume
		///
		/// \param memoryFactory the memory factory to use when creating new memory areas
		/// \param nx the number of cells in x direction
		/// \param ny the number of cells in y direction
		/// \param nz the number of cells in z direction
		/// \param numberOfGhostCells the number of ghostcells to use
		///
		EulerExtraVolume(alsfvm::shared_ptr<memory::MemoryFactory> memoryFactory,
			size_t nx, size_t ny, size_t nz, size_t numberOfGhostCells);

		///
		/// Gets the memory area representing \f$p\f$. (pressure).
		///
		ScalarMemoryPtr getP();

		///
		/// Gets the memory area representing \f$p\f$. (pressure). Const version.
		///
		ConstScalarMemoryPtr getP() const;


		///
		/// Gets the memory area representing \f$u_x\f$. (x component of velocity)
		///
		ScalarMemoryPtr getUx();

		///
		/// Gets the memory area representing \f$u_x\f$. (x component of velocity). Const version
		///
		ConstScalarMemoryPtr getUx() const;

		///
		/// Gets the memory area representing \f$u_y\f$. (y component of velocity)
		///
		ScalarMemoryPtr getUy();

		///
		/// Gets the memory area representing \f$u_y\f$. (y component of velocity). Const version
		///
		ConstScalarMemoryPtr getUy() const;

		///
		/// Gets the memory area representing \f$u_z\f$. (z component of velocity)
		///
		ScalarMemoryPtr getUz();

		///
		/// Gets the memory area representing \f$u_z\f$. (z component of velocity). Const version
		///
		ConstScalarMemoryPtr getUz() const;

    };
} // namespace alsfvm
} // namespace volume
