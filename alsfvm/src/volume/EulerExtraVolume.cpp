#include "alsfvm/volume/EulerExtraVolume.hpp"

namespace alsfvm { namespace volume { 

	// This is just to keep track of the variables
	namespace {
		enum {
			PRESSURE, UX, UY, UZ
		};
	}
	//
	/// Constructs the EulerVolume
	///
	/// \param memoryFactory the memory factory to use when creating new memory areas
	/// \param nx the number of cells in x direction
	/// \param ny the number of cells in y direction
	/// \param nz the number of cells in z direction
	///
	EulerExtraVolume::EulerExtraVolume(alsfvm::shared_ptr<memory::MemoryFactory> memoryFactory,
		size_t nx, size_t ny, size_t nz, size_t numberOfGhostCells)
		: Volume({ "p", "ux", "uy", "uz" }, memoryFactory, nx, ny, nz, numberOfGhostCells)
	{

	}

	///
	/// Gets the memory area representing \f$p\f$.
	///
	EulerExtraVolume::ScalarMemoryPtr EulerExtraVolume::getP() {
		return this->getScalarMemoryArea(PRESSURE);
	}

	///
	/// Gets the memory area representing \f$P\f$. Const version.
	///
	EulerExtraVolume::ConstScalarMemoryPtr EulerExtraVolume::getP() const {
		return this->getScalarMemoryArea(PRESSURE);
	}


	///
	/// Gets the memory area representing \f$u_x\f$. (x component of velocity)
	///
	EulerExtraVolume::ScalarMemoryPtr EulerExtraVolume::getUx() {
		return this->getScalarMemoryArea(UX);
	}

	///
	/// Gets the memory area representing \f$u_x\f$. (x component of velocity). Const version
	///
	EulerExtraVolume::ConstScalarMemoryPtr EulerExtraVolume::getUx() const {
		return this->getScalarMemoryArea(UX);
	}

	///
	/// Gets the memory area representing \f$u_y\f$. (y component of velocity)
	///
	EulerExtraVolume::ScalarMemoryPtr EulerExtraVolume::getUy() {
		return this->getScalarMemoryArea(UY);
	}

	///
	/// Gets the memory area representing \f$u_y\f$. (y component of velocity). Const version
	///
	EulerExtraVolume::ConstScalarMemoryPtr EulerExtraVolume::getUy() const {
		return this->getScalarMemoryArea(UY);
	}

	///
	/// Gets the memory area representing \f$u_z\f$. (z component of velocity)
	///
	EulerExtraVolume::ScalarMemoryPtr EulerExtraVolume::getUz() {
		return this->getScalarMemoryArea(UZ);
	}
	///
	/// Gets the memory area representing \f$u_z\f$. (z component of velocity). Const version
	///
	EulerExtraVolume::ConstScalarMemoryPtr EulerExtraVolume::getUz() const {
		return this->getScalarMemoryArea(UZ);
	}
}
}
