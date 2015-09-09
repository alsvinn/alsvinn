#pragma once
#include "alsfvm/types.hpp"

namespace alsfvm { namespace simulator { 

    class SimulatorParameters {
    public:

        real getCFLNumber() const;

    private:
        real cflNumber;

    };
} // namespace alsfvm
} // namespace simulator
