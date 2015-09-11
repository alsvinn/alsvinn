#include "alsfvm/simulator/SimulatorParameters.hpp"

namespace alsfvm { namespace simulator {

void SimulatorParameters::setCFLNumber(real cfl)
{
    cflNumber = cfl;
}

real SimulatorParameters::getCFLNumber() const
{
    return cflNumber;
}

}
}
