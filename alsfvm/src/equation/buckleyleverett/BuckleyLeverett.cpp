#include "alsfvm/equation/buckleyleverett/BuckleyLeverett.hpp"

namespace alsfvm { namespace equation { namespace buckleyleverett { 

const std::string BuckleyLeverett::name = "buckleyleverett";
const std::vector<std::string> BuckleyLeverett::conservedVariables = { "u"};



const std::vector<std::string> BuckleyLeverett::primitiveVariables = { "u" };

// Yes, empty
const std::vector<std::string> BuckleyLeverett::extraVariables;

}
}
}
