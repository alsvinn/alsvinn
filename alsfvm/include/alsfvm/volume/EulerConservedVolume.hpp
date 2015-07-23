#pragma once
#include "alsfvm/volume/Volume.hpp"


namespace alsfvm { namespace volume { 

	///
	/// Holds all the conservedvariables for an Euler simulation.
	///
	/// The conserved variables are
	/// \f[U = \left(\begin{array}{c} \rho\\ m_x\\m_y\\m_z\\E\end{array}\right)\f]
	/// where \f$\rho\f$ is the density, \f$m_x\f$, \f$m_y\f$ and \f$m_z\f$ 
	/// is the momentum in 
	/// \f$x, y, z\f$-direction, and \f$E\f$ is the energy.
	/// 
    class EulerConservedVolume : public Volume {
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
		///
		EulerConservedVolume(std::shared_ptr<memory::MemoryFactory> memoryFactory,
			size_t nx, size_t ny, size_t nz);

		///
		/// Gets the memory area representing \f$\rho\f$.
		///
		ScalarMemoryPtr getRho();

		///
		/// Gets the memory area representing \f$\rho\f$. Const version.
		///
		ConstScalarMemoryPtr getRho() const;


		///
		/// Gets the memory area representing \f$m_x\f$. (x component of momentum)
		///
		ScalarMemoryPtr getMx();

		///
		/// Gets the memory area representing \f$m_x\f$. (x component of momentum). Const version
		///
		ConstScalarMemoryPtr getMx() const;

		///
		/// Gets the memory area representing \f$m_y\f$. (y component of momentum)
		///
		ScalarMemoryPtr getMy();

		///
		/// Gets the memory area representing \f$m_y\f$. (y component of momentum). Const version
		///
		ConstScalarMemoryPtr getMy() const;

		///
		/// Gets the memory area representing \f$m_z\f$. (z component of momentum)
		///
		ScalarMemoryPtr getMz();

		///
		/// Gets the memory area representing \f$m_z\f$. (z component of momentum). Const version
		///
		ConstScalarMemoryPtr getMz() const;

		///
		/// Gets the memory area representing \f$E\f$. (Energy)
		///
		ScalarMemoryPtr getE();

		///
		/// Gets the memory area representing \f$E\f$. (Energy). Const version
		///
		ConstScalarMemoryPtr getE() const;

    };
} // namespace alsfvm
} // namespace volume
