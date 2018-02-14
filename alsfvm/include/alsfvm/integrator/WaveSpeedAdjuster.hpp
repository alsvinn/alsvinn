#pragma once
#include "alsfvm/types.hpp"

namespace alsfvm {
namespace integrator {

//! Lets observers adjust the wavespeed (this is used for eg. MPI to take the maximum
//! over all cells)
class WaveSpeedAdjuster {
    public:
        virtual ~WaveSpeedAdjuster() {};
        virtual real adjustWaveSpeed(real waveSpeed)  = 0;
};

typedef alsfvm::shared_ptr<WaveSpeedAdjuster> WaveSpeedAdjusterPtr;
} // namespace integrator
} // namespace alsfvm
