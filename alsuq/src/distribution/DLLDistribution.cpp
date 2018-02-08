#include "alsuq/distribution/DLLDistribution.hpp"
#include <boost/algorithm/string.hpp>
#include <boost/dll.hpp>

namespace alsuq { namespace distribution {


DLLDistribution::DLLDistribution(size_t numberOfSamples, size_t dimension, const Parameters &parameters)
    : size(numberOfSamples), dimension(dimension), samples(dimension, 0)
{
    auto filename = parameters.getString("library");
    auto createFunctionName = parameters.getString("create_function");

    if (boost::algorithm::to_lower_copy(createFunctionName) != "none") {
        auto createFunction = boost::dll::import<void*(int, int)>(filename, createFunctionName);
        dllData = createFunction(size, dimension);

        auto deleteFunctionName = parameters.getString("delete_function");
        if (boost::algorithm::to_lower_copy(deleteFunctionName) != "none") {
            deleteFunction = boost::dll::import<void(void*)>(filename, deleteFunctionName);
        }
    }

    auto generatorFunctionName = parameters.getString("generator_function");
    generatorFunction = boost::dll::import<real(void*, int, int, int, int)>(filename, generatorFunctionName);
}

DLLDistribution::~DLLDistribution()
{
    if (deleteFunction) {
        deleteFunction(dllData);
    }
}

real DLLDistribution::generate(generator::Generator &generator, size_t component)
{
    return generatorFunction(dllData, size, dimension, component, samples[component]++);
}

}



}
