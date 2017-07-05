#include "alsuq/run/SimulatorCreator.hpp"
#include "alsfvm/config/SimulatorSetup.hpp"
#include "alsuq/io/MPIWriterFactory.hpp"
#include "alsuq/mpi/Config.hpp"
#include "alsuq/mpi/utils.hpp"

namespace alsuq { namespace run {

SimulatorCreator::SimulatorCreator(const std::string &configurationFile,
                                   const std::vector<size_t>& samples,
                                   mpi::Config& mpiConfig)
    : mpiConfig(mpiConfig), filename(configurationFile),
      mpiCommunicator(mpiConfig.getCommunicator()),
      mpiInfo(mpiConfig.getInfo())
{

}

alsfvm::shared_ptr<alsfvm::simulator::Simulator>
SimulatorCreator::createSimulator(const alsfvm::init::Parameters &initialDataParameters,
                                                                                   size_t sampleNumber)
{

    auto groupNames = makeGroupNames(sampleNumber);
    std::shared_ptr<alsfvm::io::WriterFactory> writerFactory(
                new io::MPIWriterFactory(groupNames, mpiConfig.getRank(), firstCall, mpiCommunicator, mpiInfo));

    firstCall = false;
    alsfvm::config::SimulatorSetup simulatorSetup;
    simulatorSetup.setWriterFactory(writerFactory);
    auto simulatorPair = simulatorSetup.readSetupFromFile(filename);

    auto simulator = simulatorPair.first;
    auto initialData = simulatorPair.second;

    initialData->setParameters(initialDataParameters);

    simulator->setInitialValue(initialData);
    return simulator;
}

std::vector<std::string> SimulatorCreator::makeGroupNames(size_t sampleNumber)
{
    std::vector<size_t> samples(mpiConfig.getNumberOfProcesses());

    MPI_SAFE_CALL(MPI_Allgather((void*)&sampleNumber, 1, MPI_LONG_LONG_INT, (void*) samples.data(), 1,
                                MPI_LONG_LONG_INT, mpiCommunicator));

    std::vector<std::string> groupNames;
    groupNames.reserve(samples.size());

    for (size_t sample : samples) {
        groupNames.push_back("sample_" + std::to_string(sample));
    }

    return groupNames;
}


}
}
