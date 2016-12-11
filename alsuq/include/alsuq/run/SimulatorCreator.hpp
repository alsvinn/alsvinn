#pragma once
#include "alsfvm/simulator/Simulator.hpp"
#include "alsfvm/init/Parameters.hpp"
#include "alsuq/mpi/Config.hpp"
#include <mpi.h>

namespace alsuq { namespace run { 
//!
//! \brief The SimulatorCreator class creates a new instance of the FVM simulator
//!
    class SimulatorCreator {
    public:
        SimulatorCreator(const std::string& configurationFile,
                         const std::vector<size_t>& samples,
                         mpi::Config& config
                         );

        alsfvm::shared_ptr<alsfvm::simulator::Simulator>
        createSimulator(const alsfvm::init::Parameters& initialDataParameters,
                        size_t sampleNumber);

    private:
        mpi::Config mpiConfig;
        //! Gathers all the current samples from all current mpi procs
        //! and creates a list of names of the samples now being computed
        std::vector<std::string> makeGroupNames(size_t sampleNumber);

        bool firstCall{true};
        const std::string filename;
        MPI_Comm mpiCommunicator;
        MPI_Info mpiInfo;


    };
} // namespace run
} // namespace alsuq
