#pragma once
#include <string>
#include <boost/chrono.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

namespace alsutils {

//! Writes a run report to the json file
//!
//!    executable_name_report.json
//!
//! and to the xml file
//!
//!     executable_name_report.xml
//!
void writeRunReport(const std::string& executable,
                    const std::string& name,
                    const int cpuDurationMs,
                    const int wall,
                    const int argc,
                    char** argv);
}
