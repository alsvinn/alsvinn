#pragma once
#include "alsfvm/simulator/Simulator.hpp"
#include "alsfvm/init/Parameters.hpp"
#include <mpi.h>

namespace alsuq { namespace run { 
//!
//! \brief The SimulatorCreator class creates a new instance of the FVM simulator
//!
    class SimulatorCreator {
    public:
        SimulatorCreator(const std::string& configurationFile,
                         const std::vector<size_t>& samples,
                         MPI_Comm mpiCommunicator,
                         MPI_Info mpiInfo
                         );

        alsfvm::shared_ptr<alsfvm::simulator::Simulator>
        createSimulator(const alsfvm::init::Parameters& initialDataParameters,
                        size_t sampleNumber);

    private:
        const std::string filename;
        MPI_Comm mpiCommunicator;
        MPI_Info mpiInfo;

        std::vector<std::string> groupNames;
    };
} // namespace run
} // namespace alsuq
