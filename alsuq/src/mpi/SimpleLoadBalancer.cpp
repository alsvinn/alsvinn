#include "alsuq/mpi/SimpleLoadBalancer.hpp"
#include "alsutils/error/Exception.hpp"

namespace alsuq { namespace mpi {

SimpleLoadBalancer::SimpleLoadBalancer(const std::vector<size_t> samples)
    : samples(samples)
{

}

std::vector<size_t> SimpleLoadBalancer::getSamples(const Config &mpiConfig)
{

    size_t totalNumberOfSamples = samples.size();

    size_t numberOfProcesses = mpiConfig.getNumberOfProcesses();

    if (totalNumberOfSamples % numberOfProcesses != 0) {
        THROW("The number of processors must be a divisor of the numberOfSamples.\n"
              << "ie. totalNumberOfSamples = N * numberOfProcesses     for some integer N"
              << "\n\n"
              << "We were given:"
              << "\n\ttotalNumberOfSamples = " << totalNumberOfSamples
              << "\n\tnumberOfProcesses = " << numberOfProcesses
              <<"\n\nThis can be changed in the config file by editing"
              <<"\n\t<samples>NUMBER OF SAMPLES</samples>"
              <<"\n\n"
              <<"and as a parameter to mpirun (-np) eg."
              <<"\n\tmpirun -np 48 alsuq <path to xml>"
              <<"\n\n\nNOTE:This is a strict requirement, otherwise we have to start"
              << "\nreconfiguring the communicator when the last process runs out of samples.");
    }

    size_t rank = mpiConfig.getRank();


    size_t numberOfSamplesPerProcess = (totalNumberOfSamples) / numberOfProcesses;


    std::vector<size_t> samplesForProcess;
    samplesForProcess.reserve(numberOfSamplesPerProcess);
    for (size_t i = numberOfSamplesPerProcess*rank; i < numberOfSamplesPerProcess*(rank+1);
         ++i) {

        if (i >= totalNumberOfSamples) {
            THROW("Something went wrong in load balancing."
                  << "\nThe parameters were\n"
                  << "\n\ttotalNumberOfSamples = " << totalNumberOfSamples
                  << "\n\tnumberOfProcesses = " << numberOfProcesses
                  << "\n\tnumberOfSamplesPerProcess = " << numberOfSamplesPerProcess
                  << "\n\trank = " << rank
                  << "\n\ti= " << i);
        }
        samplesForProcess.push_back(samples[i]);
    }

    return samplesForProcess;
}

}
}
