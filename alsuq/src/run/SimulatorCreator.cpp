#include "alsuq/run/SimulatorCreator.hpp"
#include "alsfvm/config/SimulatorSetup.hpp"
#include "alsuq/io/MPIWriterFactory.hpp"
namespace alsuq { namespace run {

SimulatorCreator::SimulatorCreator(const std::string &configurationFile,
                                   const std::vector<size_t>& samples,
                                   MPI_Comm mpiCommunicator,
                                   MPI_Info mpiInfo)
    : filename(configurationFile), mpiCommunicator(mpiCommunicator),
      mpiInfo(mpiInfo)
{

    for (size_t sample : samples) {
        groupNames.push_back("sample_" + std::to_string(sample));
    }
}

alsfvm::shared_ptr<alsfvm::simulator::Simulator>
SimulatorCreator::createSimulator(const alsfvm::init::Parameters &initialDataParameters,
                                                                                   size_t sampleNumber)
{
    std::shared_ptr<alsfvm::io::WriterFactory> writerFactory(new io::MPIWriterFactory(groupNames, sampleNumber, mpiCommunicator, mpiInfo));
    alsfvm::config::SimulatorSetup simulatorSetup;
    simulatorSetup.setWriterFactory(writerFactory);
    auto simulatorPair = simulatorSetup.readSetupFromFile(filename);

    auto simulator = simulatorPair.first;
    auto initialData = simulatorPair.second;

    initialData->setParameters(initialDataParameters);

    simulator->setInitialValue(initialData);
    return simulator;
}


}
}
