#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/error/Exception.hpp"
#include "alsfvm/volume/EulerConservedVolume.hpp"
#include "alsfvm/volume/EulerExtraVolume.hpp"
#include "alsfvm/volume/EulerPrimitiveVolume.hpp"

namespace alsfvm { namespace volume { 
	/// 
	/// Constructs the factory.
	/// \param equation the equation name ("euler", "sw", etc.)
	/// \param memoryFactory the memory factory to use
	///
	VolumeFactory::VolumeFactory(const std::string& equation,
		std::shared_ptr<memory::MemoryFactory>& memoryFactory) 
		: equation(equation), memoryFactory(memoryFactory)
	{

	}


	///
	/// Creates a new volume containing the conserved variables.
	/// \param nx the number of cells in x direction
	/// \param ny the number of cells in y direction
	/// \param nz the number of cells in z direction
	///
	std::shared_ptr<Volume> VolumeFactory::createConservedVolume(size_t nx, size_t ny, size_t nz, size_t numberOfGhostCells) {
		if (equation == "euler") {
			return std::shared_ptr<Volume>(new EulerConservedVolume(memoryFactory, nx, ny, nz, numberOfGhostCells));
		}
		else {
			THROW("Unknown equation " << equation);
		}
	}

	///
	/// Creates a new volume containing the extra variables.
	/// \param nx the number of cells in x direction
	/// \param ny the number of cells in y direction
	/// \param nz the number of cells in z direction
	///
	std::shared_ptr<Volume> VolumeFactory::createExtraVolume(size_t nx, size_t ny, size_t nz, size_t numberOfGhostCells) {
		if (equation == "euler") {
			return std::shared_ptr<Volume>(new EulerExtraVolume(memoryFactory, nx, ny, nz, numberOfGhostCells));
		}
		else {
			THROW("Unknown equation " << equation);
        }
    }

    std::shared_ptr<Volume> VolumeFactory::createPrimitiveVolume(size_t nx, size_t ny, size_t nz, size_t numberOfGhostCells)
    {
        if (equation == "euler") {
            return std::shared_ptr<Volume>(new EulerPrimitiveVolume(memoryFactory, nx, ny, nz, numberOfGhostCells));
        }
        else {
            THROW("Unknown equation " << equation);
        }
    }
}
}
