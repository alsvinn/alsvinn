#pragma once
#include <string>
#include "alsfvm/boundary/Boundary.hpp"
#include "alsfvm/DeviceConfiguration.hpp"
namespace alsfvm {
namespace boundary {

class BoundaryFactory {
    public:
        ///
        /// Instantiates the boundary factory
        /// \param name the name of the boundary type
        /// Parameter | Description
        /// ----------|------------
        /// "neumann"   | Neumann boundary conditions
        /// "periodic"  | Periodic boundary conditions
        ///
        /// \param deviceConfiguration the device configuration
        ///
        BoundaryFactory(const std::string& name,
            alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration);

        ///
        /// Creates the new boundary
        /// \param ghostCellSize the number of ghost cell to use on each side.
        ///
        alsfvm::shared_ptr<Boundary> createBoundary(size_t ghostCellSize);

    private:
        std::string name;
        alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration;
};
} // namespace alsfvm
} // namespace boundary
