#include "alsuq/generator/Normal.hpp"
namespace alsuq { namespace generator {
real Normal::generate(size_t component)
{
    return distribution(generator);
}

Normal::Normal(const Parameters& parameters)
    : generator(randomDevice()),
      distribution(parameters.getParameter("mean"), parameters.getParameter("sd"))
{

}
}}
