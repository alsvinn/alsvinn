#include "alsuq/mpi/SimpleLoadBalancer.hpp"

namespace alsuq { namespace mpi {

SimpleLoadBalancer::SimpleLoadBalancer(const std::vector<size_t> samples)
    : samples(samples)
{

}

std::vector<size_t> SimpleLoadBalancer::getSamples(const Config &mpiConfig)
{
    size_t totalNumberOfSamples = samples.size();

    size_t numberOfProcesses = mpiConfig.getNumberOfProcesses();

    size_t rank = mpiConfig.getRank();


    size_t numberOfSamplesPerProcess = (totalNumberOfSamples + numberOfProcesses - 1) / numberOfProcesses;

    // take into account that the last process could have less samples than the rest if
    // the number of cores do not divide the number of samples
    size_t numberOfSamplesForThisProcess = std::min(numberOfSamplesPerProcess,
                                                    totalNumberOfSamples - rank*numberOfSamplesPerProcess);


    std::vector<size_t> samplesForProcess;
    samplesForProcess.reservice(numberOfSamplesForThisProcess);
    for (size_t i = numberOfSamplesPerProcessr*rank; i < numberOfSamplesPerProcess*rank + numberOfSamplesForThisProcess;
         ++i) {
        samplesForProcess.push_back(samples[i]);
    }

    return samplesForProcess;
}

}
}
