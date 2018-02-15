#include "alsuq/mpi/SimpleLoadBalancer.hpp"
#include "alsutils/error/Exception.hpp"

namespace alsuq {
namespace mpi {

SimpleLoadBalancer::SimpleLoadBalancer(const std::vector<size_t>& samples)
    : samples(samples) {

}

std::tuple<std::vector<size_t>,
ConfigurationPtr, ConfigurationPtr> SimpleLoadBalancer::loadBalance(
    int multiSample, ivec3 multiSpatial, const Configuration& mpiConfig) {
    // Before going through this method, read a basic tutorial on mpi communicators,
    // eg http://mpitutorial.com/tutorials/introduction-to-groups-and-communicators/
    size_t totalNumberOfSamples = samples.size();

    size_t totalNumberOfProcesses = mpiConfig.getNumberOfProcesses();

    if (multiSample * multiSpatial.x * multiSpatial.y  * multiSpatial.z !=
        int(totalNumberOfProcesses)) {
        THROW("The number of processors given (" << totalNumberOfProcesses
            << ") does not match the distribution of the samples / spatial dimensions. We were given:\n"
            << "\tmultiSample: " << multiSample << "\n"
            << "\tmultiSpatial: " << multiSpatial << "\n\n"
            << "We require that\n"
            "\tmultiSample * multiSpatial.x * multiSpatial.y  * multiSpatial.z == totalNumberOfProcesses");
    }

    if (totalNumberOfSamples % multiSample != 0) {
        THROW("The number of processors must be a divisor of the numberOfSamples.\n"
            << "ie. totalNumberOfSamples = N * numberOfProcesses     for some integer N"
            << "\n\n"
            << "We were given:"
            << "\n\ttotalNumberOfSamples = " << totalNumberOfSamples
            << "\n\tmultiSample = " << multiSample
            << "\n\nThis can be changed in the config file by editing"
            << "\n\t<samples>NUMBER OF SAMPLES</samples>"
            << "\n\n"
            << "and as a parameter to mpirun (-np) eg."
            << "\n\tmpirun -np 48 alsuq <path to xml>"
            << "\n\n\nNOTE:This is a strict requirement, otherwise we have to start"
            << "\nreconfiguring the communicator when the last process runs out of samples.");
    }


    int globalRank = mpiConfig.getRank();
    size_t numberOfSamplesPerProcess = (totalNumberOfSamples) / multiSample;

    int numberOfProcessorsPerSample = multiSpatial.x * multiSpatial.y *
        multiSpatial.z;

    const int statisticalRank = globalRank / numberOfProcessorsPerSample;
    const int spatialRank = globalRank % numberOfProcessorsPerSample;

    auto statisticalConfiguration = mpiConfig.makeSubConfiguration(spatialRank,
            statisticalRank);
    auto spatialConfiguration = mpiConfig.makeSubConfiguration(statisticalRank,
            spatialRank);

    int rank = statisticalConfiguration->getRank();
    std::vector<size_t> samplesForProcess;
    samplesForProcess.reserve(numberOfSamplesPerProcess);


    for (size_t i = numberOfSamplesPerProcess * rank;
        i < numberOfSamplesPerProcess * (rank + 1);
        ++i) {

        if (i >= totalNumberOfSamples) {
            THROW("Something went wrong in load balancing."
                << "\nThe parameters were\n"
                << "\n\ttotalNumberOfSamples = " << totalNumberOfSamples
                << "\n\tnumberOfProcesses = " << totalNumberOfProcesses
                << "\n\tnumberOfSamplesPerProcess = " << numberOfSamplesPerProcess
                << "\n\trank = " << rank
                << "\n\ti= " << i);
        }

        samplesForProcess.push_back(samples[i]);
    }

    return std::make_tuple(samplesForProcess, statisticalConfiguration,
            spatialConfiguration);
}

}
}
