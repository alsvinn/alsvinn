#pragma once
#include "alsuq/generator/Generator.hpp"
#include <random>

namespace alsuq {
namespace generator {

//! Uses the C++ STL implementation to generate random numbers
class STLMersenne : public Generator {
public:
    //! Gets the one instance of the STLMersenne generator
    static std::shared_ptr<Generator> getInstance();

    //! Generates the next random number
    real generate(size_t component);
private:
    // Singleton
    STLMersenne() {}

    std::mt19937_64 generator;
    std::uniform_real_distribution<real> distribution{0.0, 1.0};
};
} // namespace generator
} // namespace alsuq
