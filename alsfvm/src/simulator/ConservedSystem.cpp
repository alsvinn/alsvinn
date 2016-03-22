#include "alsfvm/simulator/ConservedSystem.hpp"

namespace alsfvm { namespace simulator {

ConservedSystem::ConservedSystem(alsfvm::shared_ptr<numflux::NumericalFlux> &numericalFlux)
    : numericalFlux(numericalFlux)
{

}

void ConservedSystem::operator()(const volume::Volume &conservedVariables,
                                 rvec3 &waveSpeed,
                                 bool computeWaveSpeed,
                                 volume::Volume &output)
{
    numericalFlux->computeFlux(conservedVariables, waveSpeed, computeWaveSpeed,
                               output);
}

}
}
