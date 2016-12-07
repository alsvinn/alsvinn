#include "alsuq/generator/Uniform.hpp"

namespace alsuq { namespace generator {

Uniform::Uniform(const Parameters &parameters)
    : distribution(parameters.getParameter("lower"),
                   parameters.getParameter("upper"))
{

}

real Uniform::generate(size_t component)
{
    return distribution(generator);
}

}
}
