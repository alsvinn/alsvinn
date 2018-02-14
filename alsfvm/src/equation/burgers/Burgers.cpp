#include "alsfvm/equation/burgers/Burgers.hpp"

namespace alsfvm {
namespace equation {
namespace burgers {
const std::string Burgers::name = "burgers";
const std::vector<std::string> Burgers::conservedVariables = { "u"};



const std::vector<std::string> Burgers::primitiveVariables = { "u" };

// Yes, empty
const std::vector<std::string> Burgers::extraVariables;
}
}
}
