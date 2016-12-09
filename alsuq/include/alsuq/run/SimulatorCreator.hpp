#pragma once
#include "alsfvm/simulator/Simulator.hpp"
#include "alsfvm/init/Parameters.hpp"

namespace alsuq { namespace run { 
//!
//! \brief The SimulatorCreator class creates a new instance of the FVM simulator
//!
    class SimulatorCreator {
    public:
        SimulatorCreator(const std::string& configurationFile);

        alsfvm::shared_ptr<alsfvm::simulator::Simulator>
        createSimulator(const alsfvm::init::Parameters& initialDataParameters);

    private:
        const std::string filename;
    };
} // namespace run
} // namespace alsuq
