#include "alsfvm/reconstruction/tecno/ReconstructionFactory.hpp"
#include "alsfvm/reconstruction/tecno/ENOCPU.hpp"
#include "alsfvm/reconstruction/tecno/NoReconstruction.hpp"

namespace alsfvm { namespace reconstruction { namespace tecno { 
     alsfvm::shared_ptr<TecnoReconstruction>
     ReconstructionFactory::createReconstruction(const std::string &name,
                                               const std::string &equation,
                                               const simulator::SimulatorParameters& simulatorParameters,
                                               alsfvm::shared_ptr<memory::MemoryFactory>& memoryFactory,
                                               const grid::Grid& grid,
                                               alsfvm::shared_ptr<DeviceConfiguration> &deviceConfiguration
                                               ) {

         auto platform = deviceConfiguration->getPlatform();
         alsfvm::shared_ptr<TecnoReconstruction> reconstructor;
         if (platform == "cpu") {
             if (name == "none") {
                 reconstructor.reset(new NoReconstruction);
             }
             else if (name == "eno2") {
                 reconstructor.reset(new ENOCPU<2>(memoryFactory, grid.getDimensions().x,
                     grid.getDimensions().y,
                     grid.getDimensions().z));

             }
             else if (name == "eno3") {
                 reconstructor.reset(new ENOCPU<3>(memoryFactory, grid.getDimensions().x,
                     grid.getDimensions().y,
                     grid.getDimensions().z));

             }
             else if (name == "eno4") {
                 reconstructor.reset(new ENOCPU<4>(memoryFactory, grid.getDimensions().x,
                     grid.getDimensions().y,
                     grid.getDimensions().z));

             }
         } else {
             THROW("Unknown platform " << platform);
         }


         if (!reconstructor.get()) {
             THROW("Something went wrong. Parameters were:\n"
                   << "platfrom = " << platform << "\n"
                   <<"name = " << name << "\n");
         }
         return reconstructor;

     }

}
}
}
