#include "alsfvm/volume/Volume.hpp"


#include "alsutils/error/Exception.hpp"
#include "alsfvm/volume/interpolate.hpp"
#define CHECK_SIZE_THIS(x) { \
    if (this->getNumberOfVariables() != x.getNumberOfVariables()) { \
        THROW("Number of variables do not match this, other volume: " << #x); \
    } \
    if (this->getTotalNumberOfXCells() != x.getTotalNumberOfXCells()) { \
        THROW("Not matching number of X cells. Got \n\tthis.nx = " << this->getTotalNumberOfXCells() \
               << "\n\t"<<#x<<".nx = " << x.getTotalNumberOfXCells()); \
    } \
    if (this->getTotalNumberOfYCells() != x.getTotalNumberOfYCells()) { \
        THROW("Not matching number of X cells. Got \n\tthis.ny = " << this->getTotalNumberOfYCells() \
               << "\n\t"<<#x<<".ny = " << x.getTotalNumberOfYCells()); \
    } \
    if (this->getTotalNumberOfZCells() != x.getTotalNumberOfZCells()) { \
        THROW("Not matching number of X cells. Got \n\tthis.nz = " << this->getTotalNumberOfZCells() \
               << "\n\t"<<#x<<".nz = " << x.getTotalNumberOfZCells()); \
    } \
}


namespace alsfvm {
    namespace volume {
        namespace {
            std::vector<std::string> makeComponentNameVector(const Volume& volume,
                const std::vector<size_t>& components) {
                std::vector<std::string> names;

                for (size_t i : components) {
                    names.push_back(volume.getName(i));
                }
                return names;
            }
        }
        Volume::Volume(const std::vector<std::string>& variableNames,
            alsfvm::shared_ptr<memory::MemoryFactory> memoryFactory,
            size_t nx, size_t ny, size_t nz,
            size_t numberOfGhostCells)
            :  variableNames(variableNames),
              memoryFactory(memoryFactory),
              nx(nx), ny(ny), nz(nz),
            numberOfXGhostCells(numberOfGhostCells),
            numberOfYGhostCells(ny > 1 ? numberOfGhostCells : 0),
            numberOfZGhostCells(nz > 1 ? numberOfGhostCells : 0)
        {
            for (size_t i = 0; i < variableNames.size(); i++) {
                memoryAreas.push_back(memoryFactory->createScalarMemory(
                    nx + 2 * numberOfXGhostCells,
                    ny + 2 * numberOfYGhostCells,
                    nz + 2 * numberOfZGhostCells));
            }
        }

        Volume::Volume(Volume& volume, const std::vector<size_t>& components,
            const std::vector<std::string>& variableNames)
            : variableNames(variableNames),
              memoryFactory(volume.memoryFactory),
            nx(volume.nx), ny(volume.ny), nz(volume.nz),
            numberOfXGhostCells(volume.numberOfXGhostCells),
            numberOfYGhostCells(volume.numberOfYGhostCells),
            numberOfZGhostCells(volume.numberOfZGhostCells)
        {
            for (size_t component : components) {
                memoryAreas.push_back(volume.memoryAreas[component]);
            }

        }


        Volume::~Volume()
        {
            // Everything is deleted automatically
        }

        size_t Volume::getNumberOfVariables() const {
            return variableNames.size();
        }

        ///
        /// \brief getScalarMemoryArea gets the scalar memory area (real)
        /// \param index the index of the variable. Use getIndexFromName
        ///              to get the index.
        ///
        /// \return the MemoryArea for the given index
        ///
        alsfvm::shared_ptr<memory::Memory<real> >&
            Volume::getScalarMemoryArea(size_t index) {
            return memoryAreas[index];
        }

        ///
        /// \brief getScalarMemoryArea gets the scalar memory area (real)
        /// \param index the index of the variable. Use getIndexFromName
        ///              to get the index.
        ///
        /// \return the MemoryArea for the given index
        ///
        alsfvm::shared_ptr<const memory::Memory<real> >
            Volume::getScalarMemoryArea(size_t index) const {
            return memoryAreas[index];
        }

        ///
        /// \brief getScalarMemoryArea gets the scalar memory area (real)
        /// \param name the name of the variable
        /// \return the MemoryArea for the given name
        /// \note Equivalent to calling getScalarMemoryArea(getIndexFromName(name))
        ///
        alsfvm::shared_ptr<memory::Memory<real> >&
            Volume::getScalarMemoryArea(const std::string& name) {
            return getScalarMemoryArea(getIndexFromName(name));
        }

        ///
        /// \brief getScalarMemoryArea gets the scalar memory area (real)
        /// \param name the name of the variable
        /// \return the MemoryArea for the given name
        /// \note Equivalent to calling getScalarMemoryArea(getIndexFromName(name))
        ///
        alsfvm::shared_ptr<const memory::Memory<real> >
            Volume::getScalarMemoryArea(const std::string& name) const {
            return getScalarMemoryArea(getIndexFromName(name));
        }

        ///
        /// \brief getIndexFromName returns the given index from the name
        /// \param name the name of the variable
        /// \return the index of the name.
        ///
        size_t Volume::getIndexFromName(const std::string& name) const {

            // Do this simple for now
            for (size_t i = 0; i < variableNames.size(); i++) {
                if (variableNames[i] == name) {
                    return i;
                }
            }
            THROW("Could not find variable name: " << name);
        }


        ///
        /// Gets the variable name associated to the given index
        /// \param index the index of the variable name
        /// \returns the variable name
        /// \note This implicitly uses the std::move-feature of C++11
        ///
        std::string Volume::getName(size_t index) const {
            return variableNames[index];
        }


        ///
        /// Adds each component of the other volume to this volume
        ///
        Volume& Volume::operator+=(const Volume& other) {
            for (size_t i = 0; i < memoryAreas.size(); i++) {
                (*(memoryAreas)[i]) += *(other.getScalarMemoryArea(i));
            }
            return *this;
        }


        /// 
        /// Multiplies each component of the volume by the scalar
        ///
        Volume& Volume::operator*=(real scalar) {
            for (size_t i = 0; i < memoryAreas.size(); i++) {
                (*(memoryAreas)[i]) *= scalar;
            }
            return *this;
        }

        ///
        /// \returns the number of cells in X direction
        ///
        size_t Volume::getNumberOfXCells() const {
            return nx;
        }

        ///
        /// \returns the number of cells in Y direction
        ///
        size_t Volume::getNumberOfYCells() const {
            return ny;
        }

        ///
        /// \returns the number of cells in Z direction
        ///
        size_t Volume::getNumberOfZCells() const {
            return nz;
        }

        void Volume::copyInternalCells(size_t memoryAreaIndex, real *output, size_t outputSize) const
        {
            memoryAreas[memoryAreaIndex]->copyInternalCells(numberOfXGhostCells, getTotalNumberOfXCells() - numberOfXGhostCells,
                numberOfYGhostCells, getTotalNumberOfYCells() - numberOfYGhostCells,
                numberOfZGhostCells, getTotalNumberOfZCells() - numberOfZGhostCells, output, outputSize);
        }

        void Volume::makeZero()
        {
            for (size_t i = 0; i < memoryAreas.size(); i++) {
                memoryAreas[i]->makeZero();
            }
        }

        ///
        /// Gets the number of ghost cells in x direction
        /// \note This is the number of ghost cells on one side.
        ///
        size_t Volume::getNumberOfXGhostCells() const {
            return numberOfXGhostCells;
        }

        ///
        /// Gets the number of ghost cells in y direction
        /// \note This is the number of ghost cells on one side.
        ///
        size_t Volume::getNumberOfYGhostCells() const {
            return numberOfYGhostCells;
        }

        ///
        /// Gets the number of ghost cells in z direction
        /// \note This is the number of ghost cells on one side.
        ///
        size_t Volume::getNumberOfZGhostCells() const {
            return numberOfZGhostCells;
        }

        ///
        /// Returns the total number of cells in x direction, including ghost cells
        ///
        size_t Volume::getTotalNumberOfXCells() const {
            return nx + 2 * numberOfXGhostCells;
        }

        ///
        /// Returns the total number of cells in y direction, including ghost cells
        ///
        size_t Volume::getTotalNumberOfYCells() const {
            return ny + 2 * numberOfYGhostCells;
        }

        ///
        /// Returns the total number of cells in z direction, including ghost cells
        ///
        size_t Volume::getTotalNumberOfZCells() const {
            return nz + 2 * numberOfZGhostCells;
        }

        size_t Volume::getDimensions() const {
            if (ny == 1) {
                return 1;
            }
            else if (nz == 1) {
                return 2;
            }
            else {
                return 3;
            }
        }

        /// 
        /// Copies the whole volume to the other volume
        ///
        void Volume::copyTo(volume::Volume& other) const {
            if (getNumberOfVariables() == 0) {
                return;
            }
            std::vector<real> temporaryStorage(getScalarMemoryArea(0)->getSize());
            for (size_t var = 0; var < getNumberOfVariables(); ++var) {
                getScalarMemoryArea(var)->copyToHost(temporaryStorage.data(), temporaryStorage.size());
                other.getScalarMemoryArea(var)->copyFromHost(temporaryStorage.data(), temporaryStorage.size());
            }
        }

        void Volume::setVolume(const Volume &other)
        {
            if (other.getNumberOfVariables() != getNumberOfVariables()) {
                THROW("Trying to interpolate between to volumes, but"
                    "we do not have the same number of variables.");
            }

            const size_t nxOther = other.getNumberOfXCells();
            const size_t nyOther = other.getNumberOfYCells();
            const size_t nzOther = other.getNumberOfZCells();

            // Base case.
            if (nxOther == nx && nyOther == ny && nzOther == nz) {
                other.copyTo(*this);
                return;
            }

            // Test which dimension we are in.
            if (getNumberOfYCells() == 1) {
                interpolate<1>(*this, other);
            }
            else if (getNumberOfZCells() == 1) {
                interpolate<2>(*this, other);
            }
            else {
                interpolate<3>(*this, other);
            }


        }
        
        alsfvm::shared_ptr<const memory::Memory<real> >
            Volume::operator[](size_t index) const {
            return this->getScalarMemoryArea(index);
        }

        alsfvm::shared_ptr<memory::Memory<real> > Volume::operator[](size_t index)
        {
            return getScalarMemoryArea(index);
        }

        //! Adds the volumes with coefficients to this volume
        //! Here we compute the sum
        //! \f[ v_1^{\mathrm{new}}=a_1v_1+a_2v_2+a_3v_3+a_4v_4+a_5v_5+a_6v_6\f]
        //! where \f$v_1\f$ is the volume being operated on.
        void Volume::addLinearCombination(real a1, real a2, const Volume& v2, real a3, const Volume& v3, real a4, const Volume& v4, real a5, const Volume& v5)
        {
            CHECK_SIZE_THIS(v2);
            CHECK_SIZE_THIS(v3);
            CHECK_SIZE_THIS(v4);
            CHECK_SIZE_THIS(v5);
            for (size_t i = 0; i < memoryAreas.size(); ++i) {
                this->getScalarMemoryArea(i)->addLinearCombination(a1, a2, *v2[i], a3, *v3[i], a4, *v4[i], a5, *v5[i]);
            }
        }

        ivec3 Volume::getTotalDimensions() const
        {
            return {int(getTotalNumberOfXCells()),
                    int(getTotalNumberOfYCells()),
                    int(getNumberOfZCells())};
        }

        void Volume::addPower(const Volume &other, real power)
        {
            for(size_t i = 0; i < memoryAreas.size(); ++i) {
                memoryAreas[i]->addPower(*other[i], power);
            }

        }


        void Volume::subtractPower(const Volume &other, real power)
        {
            for(size_t i = 0; i < memoryAreas.size(); ++i) {
                memoryAreas[i]->subtractPower(*other[i], power);
            }

        }


        std::shared_ptr<Volume> Volume::makeInstance() const
        {
            return std::make_shared<Volume>(variableNames, memoryFactory,
                                            nx, ny, nz, numberOfXGhostCells);
        }

        std::shared_ptr<Volume> Volume::makeInstance(size_t nxNew, size_t nyNew, size_t nzNew, const std::string& platform) const
        {
            if (platform == "default" || platform == memoryFactory->getPlatform()) {
                return std::make_shared<Volume>(variableNames, memoryFactory,
                                            nxNew, nyNew, nzNew, 0);
            } else {
                alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguraiton(new DeviceConfiguration(platform));
                alsfvm::shared_ptr<memory::MemoryFactory>
                        memoryFactoryForPlatform(new memory::MemoryFactory(deviceConfiguraiton));

                return std::make_shared<Volume>(variableNames, memoryFactoryForPlatform, nxNew, nyNew, nzNew, 0);
            }

        }
	}

}
