#include "alsfvm/functional/LogEntropy.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/functional/register_functional.hpp"
#include <iostream>
namespace alsfvm {
namespace functional {

LogEntropy::LogEntropy(const Functional::Parameters& parameters)
    : gamma(parameters.getDouble("gamma")) {


}

void LogEntropy::operator()(volume::Volume& conservedVolumeOut,
    const volume::Volume& conservedVolumeIn, const real weight,
    const grid::Grid& grid) {

    const auto lengths = grid.getCellLengths();

    const real dxdydz = lengths.x * lengths.y * lengths.z;


    real integral = 0.0;

    const auto& densityView = conservedVolumeIn.getScalarMemoryArea(
            "rho")->getView();

    const auto& energyView = conservedVolumeIn.getScalarMemoryArea(
            "E")->getView();

    const size_t numberOfComponents = grid.getActiveDimension();

    std::vector<const real*> momentumPointers;


    momentumPointers.push_back(
        conservedVolumeIn.getScalarMemoryArea("mx")->getPointer());

    if (numberOfComponents > 1) {
        momentumPointers.push_back(
            conservedVolumeIn.getScalarMemoryArea("my")->getPointer());
    }

    if (numberOfComponents > 2) {
        momentumPointers.push_back(
            conservedVolumeIn.getScalarMemoryArea("mz")->getPointer());
    }






    volume::for_each_midpoint(conservedVolumeIn, grid, [&](real, real, real,
    size_t i) {

        const real density = densityView.at(i);
        const real energy = energyView.at(i);

        double momentumSquared = 0.0;

        for (size_t component = 0; component < numberOfComponents; ++component) {
            const auto momentum = momentumPointers[component][i];
            momentumSquared += momentum * momentum;
        }

        const real pressure = (gamma - 1) * (energy - 1.0 / (2 * density) *
                momentumSquared);


        const real s = log(pressure) - gamma * log(density);
        const real E = (-density * s) / (gamma - 1);

        integral += E * dxdydz;
    });

    conservedVolumeOut.getScalarMemoryArea("E")->getPointer()[0] += weight
        * integral;
}

ivec3 LogEntropy::getFunctionalSize(const grid::Grid&) const {
    return {1, 1, 1};
}
REGISTER_FUNCTIONAL(cpu, log_entropy, LogEntropy)
}
}
