#include <vector>
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsutils/error/Exception.hpp"
#include "alsfvm/equation/equation_list.hpp"

namespace alsfvm { namespace volume { 
    namespace {
        


        //! We will loop through all equations to find the correct one 
        //! instiate for.
        struct EquationFunctor {
            enum VolumeType {
                CONSERVED, EXTRA, PRIMITIVE
            };

            EquationFunctor(const std::string& equationName,
                VolumeType type,
                alsfvm::shared_ptr<memory::MemoryFactory> memoryFactory,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t numberOfGhostCells,
                alsfvm::shared_ptr<Volume>& volumePointer)
                : equationName(equationName), 
                type(type),
                memoryFactory(memoryFactory),
                nx(nx),
                ny(ny),
                nz(nz),
                numberOfGhostCells(numberOfGhostCells),
                volumePointer(volumePointer)
            {

            }

            template<class EquationInfo>
            void operator()(const EquationInfo& info) const {
                typedef typename EquationInfo::EquationType Equation;
                
                if (info.getName() == equationName) {
                    std::vector<std::string> names;
                    switch (type) {
                    case CONSERVED:
                        names = Equation::conservedVariables;
                        break;

                    case EXTRA:
                        names = Equation::extraVariables;
                        break;

                    case PRIMITIVE:
                        names = Equation::primitiveVariables;
                        break;
                    }
                    volumePointer.reset(new Volume(names,
                        memoryFactory,
                        nx, ny, nz,
                        numberOfGhostCells));
                }

             
            }


            std::string equationName;
            VolumeType type;
            alsfvm::shared_ptr<memory::MemoryFactory> memoryFactory;
            size_t nx;
            size_t ny;
            size_t nz;
            size_t numberOfGhostCells;
            alsfvm::shared_ptr<Volume>& volumePointer;
        };
    }


	/// 
	/// Constructs the factory.
	/// \param equation the equation name ("euler1", "euler2", "euler3",  "sw", etc.)
	/// \param memoryFactory the memory factory to use
	///
	VolumeFactory::VolumeFactory(const std::string& equation,
        alsfvm::shared_ptr<memory::MemoryFactory> memoryFactory)
		: equation(equation), memoryFactory(memoryFactory)
	{

	}


	///
	/// Creates a new volume containing the conserved variables.
	/// \param nx the number of cells in x direction
	/// \param ny the number of cells in y direction
	/// \param nz the number of cells in z direction
	///
	alsfvm::shared_ptr<Volume> VolumeFactory::createConservedVolume(size_t nx, size_t ny, size_t nz, size_t numberOfGhostCells) {
        alsfvm::shared_ptr<Volume> volumePointer;
        EquationFunctor functor(equation, EquationFunctor::CONSERVED, memoryFactory, nx, ny, nz, numberOfGhostCells, volumePointer);

        equation::for_each_equation(functor);


		if (!volumePointer) {
			THROW("Unknown equation " << equation);
		}

        return volumePointer;
	}

	///
	/// Creates a new volume containing the extra variables.
	/// \param nx the number of cells in x direction
	/// \param ny the number of cells in y direction
	/// \param nz the number of cells in z direction
	///
	alsfvm::shared_ptr<Volume> VolumeFactory::createExtraVolume(size_t nx, size_t ny, size_t nz, size_t numberOfGhostCells) {
        alsfvm::shared_ptr<Volume> volumePointer;
        EquationFunctor functor(equation, EquationFunctor::EXTRA, memoryFactory, nx, ny, nz, numberOfGhostCells, volumePointer);
        equation::for_each_equation(functor);

        if (!volumePointer) {
            THROW("Unknown equation " << equation);
        }

        return volumePointer;
    }

    alsfvm::shared_ptr<Volume> VolumeFactory::createPrimitiveVolume(size_t nx, size_t ny, size_t nz, size_t numberOfGhostCells)
    {
        alsfvm::shared_ptr<Volume> volumePointer;
        EquationFunctor functor(equation, EquationFunctor::PRIMITIVE, memoryFactory, nx, ny, nz, numberOfGhostCells, volumePointer);

        equation::for_each_equation(functor);
        if (!volumePointer) {
            THROW("Unknown equation " << equation);
        }

        return volumePointer;
    }
}
}
