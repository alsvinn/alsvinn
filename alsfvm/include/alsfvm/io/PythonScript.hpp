#pragma once
#include "alsutils/config.hpp"
#include "alsfvm/io/Writer.hpp"
#include "alsfvm/io/Parameters.hpp"
#include "alsfvm/python/PythonInterpreter.hpp"
#include <boost/python.hpp>
#include "alsutils/mpi/Configuration.hpp"

namespace alsfvm {
namespace io {

//! Allows you to run a python script on the provided data
//!
//! Parameters expected:
//!
//!   * pythonCode
//!   * pythonClass
//!
class PythonScript : public io::Writer {
public:


    PythonScript(const std::string& basename,
        const Parameters& parameters,
        alsutils::mpi::ConfigurationPtr mpiConfigration = nullptr

    );
    ///
    /// \brief write writes the data to disk
    /// \param conservedVariables the conservedVariables to write
    /// \param extraVariables the extra variables to write
    /// \param grid the grid that is used (describes the _whole_ domain)
    /// \param timestepInformation
    ///
    virtual void write(const volume::Volume& conservedVariables,
        const volume::Volume& extraVariables,
        const grid::Grid& grid,
        const simulator::TimestepInformation& timestepInformation);


    //! This method should be called at the end of the simulation
    virtual void finalize(const grid::Grid& grid,
        const simulator::TimestepInformation& timestepInformation) override;

private:

    const std::string pythonCode;
    const std::string pythonClass;

    python::PythonInterpreter& pythonInterpreterInstance;

    boost::python::object mainModule;
    boost::python::object mainNamespace;

    boost::python::object classInstance;

    boost::python::dict datasetsConserved;
    boost::python::dict datasetsExtra;

    std::vector<double*> rawPointersConserved;
    std::vector<double*> rawPointersExtra;

    void makeDatasets(const volume::Volume& conservedVariables,
        const volume::Volume& extraVariables);

    void copyToDatasets( const volume::Volume& conservedVariables,
        const volume::Volume& extraVariables);

    boost::python::object makeGrid(const grid::Grid& grid);

    bool datasetsInitialized = false;

    alsutils::mpi::ConfigurationPtr mpiConfiguration;


};
} // namespace io
} // namespace alsfvm
