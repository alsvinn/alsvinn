#include "alsfvm/functional/FunctionalFactory.hpp"

namespace alsfvm {
namespace functional {
namespace {
class FunctionalList {


    public:
        static FunctionalList& instance() {
            static FunctionalList list;

            return list;
        }
        std::map<std::string, std::map<std::string, FunctionalFactory::FunctionalCreator> >
        creators;
    private:
        FunctionalList() {}

};
}

void FunctionalFactory::registerFunctional(const std::string& platform,
    const std::string& name,
    FunctionalFactory::FunctionalCreator maker) {
    auto& list = FunctionalList::instance().creators;

    if (list[platform].find(name) != list[platform].end()) {
        THROW("'" << name << "' already registered as a functional");
    }

    list[platform][name] = maker;
}

FunctionalPointer FunctionalFactory::makeFunctional(const std::string& platform,
    const std::string& name,
    const FunctionalFactory::Parameters& parameters) {
    auto& list = FunctionalList::instance().creators;

    if (list[platform].find(name) == list[platform].end()) {
        THROW("Unknown functional: " << name);
    }

    return list[platform][name](parameters);
}

}
}
