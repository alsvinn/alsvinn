#pragma once
#include "alsuq/generator/Generator.hpp"
#include "alsuq/generator/Parameters.hpp"
#include <random>

namespace alsuq { namespace generator {
class Normal : public Generator {
public:
    Normal(const Parameters& parameters);
    real generate(size_t component);

private:
    std::random_device randomDevice;
    std::mt19937 generator;


    std::normal_distribution<> distribution;
};


}}
