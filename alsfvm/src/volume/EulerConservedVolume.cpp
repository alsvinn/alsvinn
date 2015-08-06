#include "alsfvm/volume/EulerConservedVolume.hpp"

namespace alsfvm {
	namespace volume {

		// This is just to keep track of the variables
		namespace {
			enum {
				RHO, MX, MY, MZ, ENERGY
			};
		}
		///
		/// Constructs the EulerVolume
		///
		/// \param memoryFactory the memory factory to use when creating new memory areas
		/// \param nx the number of cells in x direction
		/// \param ny the number of cells in y direction
		/// \param nz the number of cells in z direction
		///
		EulerConservedVolume::EulerConservedVolume(std::shared_ptr<memory::MemoryFactory> memoryFactory,
			size_t nx, size_t ny, size_t nz, size_t numberOfGhostCells)
			: Volume({ "rho", "mx", "my", "mz", "E" }, memoryFactory, nx, ny, nz, numberOfGhostCells)
		{

		}

		///
		/// Gets the memory area representing \f$\rho\f$.
		///
		EulerConservedVolume::ScalarMemoryPtr EulerConservedVolume::getRho() {
			return this->getScalarMemoryArea(RHO);
		}

		///
		/// Gets the memory area representing \f$\rho\f$. Const version.
		///
		EulerConservedVolume::ConstScalarMemoryPtr EulerConservedVolume::getRho() const {
			return this->getScalarMemoryArea(RHO);
		}


		///
		/// Gets the memory area representing \f$m_x\f$. (x component of momentum)
		///
		EulerConservedVolume::ScalarMemoryPtr EulerConservedVolume::getMx() {
			return this->getScalarMemoryArea(MX);
		}

		///
		/// Gets the memory area representing \f$m_x\f$. (x component of momentum). Const version
		///
		EulerConservedVolume::ConstScalarMemoryPtr EulerConservedVolume::getMx() const {
			return this->getScalarMemoryArea(MX);
		}

		///
		/// Gets the memory area representing \f$m_y\f$. (y component of momentum)
		///
		EulerConservedVolume::ScalarMemoryPtr EulerConservedVolume::getMy() {
			return this->getScalarMemoryArea(MY);
		}

		///
		/// Gets the memory area representing \f$m_y\f$. (y component of momentum). Const version
		///
		EulerConservedVolume::ConstScalarMemoryPtr EulerConservedVolume::getMy() const {
			return this->getScalarMemoryArea(MY);
		}

		///
		/// Gets the memory area representing \f$m_z\f$. (z component of momentum)
		///
		EulerConservedVolume::ScalarMemoryPtr EulerConservedVolume::getMz() {
			return this->getScalarMemoryArea(MZ);
		}

		///
		/// Gets the memory area representing \f$m_z\f$. (z component of momentum). Const version
		///
		EulerConservedVolume::ConstScalarMemoryPtr EulerConservedVolume::getMz() const  {
			return this->getScalarMemoryArea(MZ);
		}

		///
		/// Gets the memory area representing \f$E\f$. (Energy)
		///
		EulerConservedVolume::ScalarMemoryPtr EulerConservedVolume::getE()  {
			return this->getScalarMemoryArea(ENERGY);
		}

		///
		/// Gets the memory area representing \f$E\f$. (Energy). Const version
		///
		EulerConservedVolume::ConstScalarMemoryPtr EulerConservedVolume::getE() const  {
			return this->getScalarMemoryArea(ENERGY);
		}
	}
}
