#include <gtest/gtest.h>
#include "alsuq/distribution/DistributionFactory.hpp"
#include "alsuq/generator/GeneratorFactory.hpp"
#include "utils/polyfit.hpp"

using namespace alsuq;
namespace {
    struct MeanVarTestParameters {
        MeanVarTestParameters(const std::string& generatorName,
                              const std::string& distName,
                              real mean, real variance,
                              std::vector<std::string> parameterNames,
                              std::vector<real> parameterValues)
            : generatorName(generatorName), distributionName(distName), mean(mean), variance(variance)
        {
            for (int i = 0; i < parameterNames.size(); ++i) {
                parameters.setParameter(parameterNames[i], parameterValues[i]);
            }
        }

        std::string generatorName;
        std::string distributionName;

        alsuq::distribution::Parameters parameters;
        real mean;
        real variance;
    };

    std::ostream& operator<<(std::ostream& os, const MeanVarTestParameters& parameters) {
        os << "\n{\n\tgeneratorName = " << parameters.generatorName
           << "\n\tdistributionName = " << parameters.distributionName
           << "\n\tmean = "<< parameters.mean
           << "\n\tvariance = " << parameters.variance
           << std::endl << "}" << std::endl;
        return os;
    }
}

struct MeanVarTest : public ::testing::TestWithParam <MeanVarTestParameters> {
    MeanVarTest () {

    }



    std::shared_ptr<alsuq::generator::Generator> makeGenerator(int M) {
        auto parameters = GetParam();
        alsuq::generator::GeneratorFactory factory;
        return factory.makeGenerator(parameters.generatorName, 1, M);
    }

    std::shared_ptr<alsuq::distribution::Distribution> makeDistribution(int M) {
        auto parameters = GetParam();
        alsuq::distribution::DistributionFactory factory;
        return factory.createDistribution(parameters.distributionName, 1, M, parameters.parameters);
    }
};



TEST_P(MeanVarTest, MeanVarTest) {
    // make sure we converge towards the mean with rate 1/2
    std::vector<real> logM;
    std::vector<real> logMeanError;
    std::vector<real> logVarError;

    const real actualVar = GetParam().variance;
    const real actualMean = GetParam().mean;
    for (int k = 4; k < 20; k++) {
        int M = 1<<k;
        auto generator = makeGenerator(M);
        auto dist = makeDistribution(M);
        real mean = 0;
        real M2 = 0;
        int N = 0;
        for (int n = 0; n < M; ++n) {
            N+=1;
            real x = dist->generate(*generator, 0);
            real delta = x - mean;
            mean += delta/N;
            real delta2 = x - mean;
            M2 += delta*delta2;
        }


        real var = M2 / (N-1);


        logM.push_back(std::log(M));
        logMeanError.push_back(std::log(std::abs(mean-actualMean)));
        logVarError.push_back(std::log(std::abs(var-actualVar)));
    }

    double rateMean = alsfvm::linearFit(logM, logMeanError)[0];

    ASSERT_LE(0.3, -rateMean);

    double rateVar= alsfvm::linearFit(logM, logVarError)[0];

    ASSERT_LE(0.3, -rateVar);
}

INSTANTIATE_TEST_CASE_P(MeanVarConvergenceTests,
    MeanVarTest,
    ::testing::Values(
    MeanVarTestParameters("well512a", "normal", 0, 1, {"mean", "sd"}, {0,1}),
    MeanVarTestParameters("well512a", "uniform", 0.5, 1.0/12., {"lower", "upper"}, {0,1}),
    MeanVarTestParameters("stlmersenne", "normal", 0, 1, {"mean", "sd"}, {0,1}),
    MeanVarTestParameters("stlmersenne", "uniform", 0.5, 1.0/12., {"lower", "upper"}, {0,1})
    ));
