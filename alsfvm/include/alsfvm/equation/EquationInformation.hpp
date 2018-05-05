#pragma once
#include <string>
namespace alsfvm {
namespace equation {

///
/// Simple holder class for equations, as an equation can not be held in a
/// tuple (there is not always a default constructor available)
///
template<class T>
class EquationInformation {
public:
    typedef T EquationType;

    ///
    /// Gets the name of the equation held in the EquationInformationType.
    ///
    static std::string getName() {

        return EquationType::getName();
    }
};
} // namespace alsfvm
} // namespace equation
