#include "alsuq/run/SimulatorCreator.hpp"
#include "alsfvm/config/SimulatorSetup.hpp"
namespace alsuq { namespace run {

SimulatorCreator::SimulatorCreator(const std::string &configurationFile)
    : filename(configurationFile)
{

}

alsfvm::shared_ptr<alsfvm::simulator::Simulator> SimulatorCreator::createSimulator(const alsfvm::init::Parameters &initialDataParameters)
{
    alsfvm::config::SimulatorSetup simulatorSetup;
    auto simulator =simulatorSetup.readSetupFromFile(filename);



    return simulator;
}


}
}
