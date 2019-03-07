#include "alsutils/debug/stacktrace.hpp"



#include <array>
#include <sstream>
#ifdef __GNUC__
    #include <execinfo.h>
    #include <cxxabi.h>
    #include <iostream>
    #include <boost/core/demangle.hpp>
    #include <regex>
    #include <boost/lexical_cast.hpp>
#endif

namespace alsutils {
namespace debug {

namespace  {

BacktraceInformation getFilenameFunctionLineNumber(
    const std::string& backtraceString) {
    std::regex regex("(.+)(\\((.+)\\+(.+)\\))", std::regex_constants::ECMAScript);
    std::smatch matches;

    if (std::regex_search(backtraceString, matches, regex)) {
        const std::string filename = matches[1];
        const std::string function = boost::core::demangle(std::string(
                    matches[3]).c_str());
        const int lineNumber =  std::stoul(matches[4], nullptr, 16);

        return BacktraceInformation{function, filename, std::to_string(lineNumber)};
    }

    return BacktraceInformation{backtraceString, "", ""};
}

std::string makeHumanReadableLong(const BacktraceInformation&
    backtraceInformation) {

    return backtraceInformation.functionName + " at line " +
        backtraceInformation.lineNumber + " (in " + backtraceInformation.filename + ")";
}


std::string makeHumanReadableShort(const BacktraceInformation&
    backtraceInformation) {

    const auto& functionName = backtraceInformation.functionName;
    std::regex regex("(([^\\(]+(::)?)+)", std::regex_constants::ECMAScript);
    std::smatch matches;

    if (std::regex_search(functionName, matches, regex)) {
        // Remove any template arguments
        std::regex regexTemplate("<.+>", std::regex_constants::ECMAScript);

        return std::regex_replace(std::string(matches[1]), regexTemplate, "");
    }

    return "";
}


}
std::vector<BacktraceInformation> getStacktrace() {
#ifdef __GNUC__
    constexpr size_t maxNumberOfLevels = 100;

    std::array<void*, maxNumberOfLevels> buffer;

    auto numberOfSymbols = backtrace(buffer.data(), maxNumberOfLevels);


    auto strings = backtrace_symbols(buffer.data(), numberOfSymbols);

    std::vector<BacktraceInformation> functionNames;

    for (decltype(numberOfSymbols) i = 1; i < numberOfSymbols; ++i) {
        functionNames.push_back(getFilenameFunctionLineNumber(strings[i]));
    }

    free(strings);

    return functionNames;
#else
    return {BacktraceInformation{"Stacktrace only supported on GCC", "", ""}};
#endif
}

std::string getShortStacktrace() {
    std::stringstream ss;
    const auto stacktrace = getStacktrace();

    for (size_t i = 1; i < stacktrace.size(); ++i) {
        ss << makeHumanReadableShort(stacktrace[i]);

        if (i < stacktrace.size() - 1) {
            ss << "-->";
        }
    }

    return ss.str();
}



std::string getLongStacktrace() {
    std::stringstream ss;
    const auto stacktrace = getStacktrace();

    for (size_t i = 1; i < stacktrace.size(); ++i) {
        ss << "\t * " << makeHumanReadableLong(stacktrace[i]);

        if (i < stacktrace.size() - 1) {
            ss << "\n";
        }
    }

    return ss.str();
}

}

}
