#pragma once
#include "alsfvm/simulator/Simulator.hpp"
#include "alsfvm/init/Parameters.hpp"
#include "alsuq/mpi/Configuration.hpp"
#include <mpi.h>
#include "alsuq/types.hpp"

namespace alsuq { namespace run { 
//!
//! \brief The SimulatorCreator class creates a new instance of the FVM simulator
//!
    class SimulatorCreator {
    public:
        SimulatorCreator(const std::string& configurationFile,
                         mpi::ConfigurationPtr mpiConfigurationSpatial,
                         mpi::ConfigurationPtr mpiConfigurationStatistical,
                         mpi::ConfigurationPtr mpiConfigurationWorld,
                         ivec3 multiSpatial
                         );

        alsfvm::shared_ptr<alsfvm::simulator::Simulator>
        createSimulator(const alsfvm::init::Parameters& initialDataParameters,
                        size_t sampleNumber);

    private:
        mpi::ConfigurationPtr mpiConfigurationSpatial;
        mpi::ConfigurationPtr mpiConfigurationStatistical;
        mpi::ConfigurationPtr mpiConfigurationWorld;

        ivec3 multiSpatial;

        //! Gathers all the current samples from all current mpi procs
        //! and creates a list of names of the samples now being computed
        std::vector<std::string> makeGroupNames(size_t sampleNumber);

        bool firstCall{true};
        const std::string filename;



    };
} // namespace run
} // namespace alsuq
