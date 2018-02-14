#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/equation/EquationParameters.hpp"
namespace alsfvm {
namespace simulator {

class SimulatorParameters {
    public:
        SimulatorParameters()
            : equationParameters(new equation::EquationParameters)
        {}
        SimulatorParameters(const std::string& equationName,
            const std::string& platform);

        void setCFLNumber(real cfl);
        real getCFLNumber() const;

        const equation::EquationParameters& getEquationParameters() const;
        equation::EquationParameters& getEquationParameters();
        void setEquationParameters(alsfvm::shared_ptr<equation::EquationParameters>
            parameters);

        void setEquationName(const std::string& name);

        const std::string& getEquationName() const;

        void setPlatform(const std::string& platform);

        const std::string& getPlatform() const;


    private:
        real cflNumber;
        std::string equationName;
        std::string platform;
        alsfvm::shared_ptr<equation::EquationParameters> equationParameters;

};
} // namespace alsfvm
} // namespace simulator
