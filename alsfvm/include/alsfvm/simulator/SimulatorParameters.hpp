#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/equation/EquationParameters.hpp"
namespace alsfvm { namespace simulator { 

    class SimulatorParameters {
    public:
        void setCFLNumber(real cfl);
        real getCFLNumber() const;


    private:
        real cflNumber;

    };
} // namespace alsfvm
} // namespace simulator
