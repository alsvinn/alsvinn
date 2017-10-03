#pragma once
#include <sstream>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/sources/global_logger_storage.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/logger.hpp>

#define INFO alsutils::log::severity_level::info
#define NOTIFICATION alsutils::log::severity_level::notification
#define WARNING alsutils::log::severity_level::warning
#define ERROR alsutils::log::severity_level::error
#define CRITICAL alsutils::log::severity_level::critical



#define ALSVINN_LOG(severity, message) { \
    std::stringstream ssForLog; \
    ssForLog << message; \
    BOOST_LOG_SEV(::alsutils::log::alsvinnLogger::get(), severity) << ssForLog.str(); \
    }

namespace alsutils {
    namespace log {

        enum severity_level
        {
            info,
            notification,
            warning,
            error,
            critical
        };


        BOOST_LOG_INLINE_GLOBAL_LOGGER_DEFAULT(alsvinnLogger, boost::log::sources::severity_logger_mt<severity_level>)


        void inline setLogFile(const std::string& filename) {
            boost::log::add_file_log(filename, boost::log::keywords::auto_flush = true );
        }
    }
}
