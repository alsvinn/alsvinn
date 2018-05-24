/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
