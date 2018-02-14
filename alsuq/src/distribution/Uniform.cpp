#include "alsuq/distribution/Uniform.hpp"

namespace alsuq {
namespace distribution {

Uniform::Uniform(const Parameters& parameters)
    : a(parameters.getParameter("lower")),
      b(parameters.getParameter("upper")) {

}

real Uniform::generate(generator::Generator& generator, size_t component) {
    return scale(generator.generate(component));
}

real Uniform::scale(real x) {
    return (x * (b - a) + a);
}

}
}
