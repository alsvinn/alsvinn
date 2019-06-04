#pragma once
#include "alsfvm/io/Writer.hpp"
#include "alsfvm/io/Parameters.hpp"
#include "alsutils/mpi/Configuration.hpp"

namespace alsfvm {
namespace io {

//! The DLLWriter forwards the volume to an external DLL, loaded via
//! boost::dll
//!
//! The parameters you need to supply are
//!
//! <table>
//! <tr><th> parameter name</th> <th>description</th></tr>
//! <tr><td>library          </td><td> filename of dll</td></tr>
//!
//! <tr><td>create_function  </td><td> name of the create/init function, should have the following signature<br />
//!                           \code{.cpp}
//!                              void* create_function(const char* simulator_name, const char* simulator_version, void* parameters);
//!                           \endcode
//!                           use NONE if it is not supplied</td></tr>
//!
//! <tr><td>delete_function </td><td> the function to delete any data created,
//!                           if create_function is NONE, this is ignored
//!                           assumes signature
//!                           \code{.cpp}
//!                              void delete_function(void* data);
//!                           \endcode</td></tr>
//!
//! <tr><td>write_function </td><td> the name of the write function
//!                            assumes signature
//!                            \code{.cpp}
//!                               void write_function(void* data, void* parameters, real time, const char* variable_name, const real* variable_data, int nx, int ny, int nz, int ngx, int ngy, int ngz, real ax, real ay, real az, real bx, real by, real bz, int gpu_number );
//!                            \endcode
//! where gpu_number is the gpu id where the data lives (-1 if the data is on the CPU).
//!
//! The domain is \f$[ax, bx] \times [ay, by] \times [az, bz]\f$ discretized into nx, ny, and nx cells, with an ADDITIONAL ngx, ngy and ngz ghost cells in each direction and side.
//!
//! The total number of cells is \f$(nx+2*ngx)*(ny+2*ngy)*(nz+2*ngz)\f$
//! </td></tr>
//!
//! <tr><td>make_parameters_function </td><td> Name of the function to create the parameter struct
//!                            assumes the signature
//!                            \code{.cpp}
//!                               void* make_parameters_function();
//!                            \endcode</td></tr>
//!
//! <tr><td>delete_parameters_function </td><td> name of the function to delete the parameter struct
//!                            assumes the signature
//!                            \code{.cpp}
//!                               void delete_parameters_function(void* parameters);
//!                            \endcode</td></tr>
//!
//! <tr><td>needs_data_on_host_function</td><td> should the data be on host? If this function returns true, alsvinn will first copy the data to host before calling the write function
//!                            \code{.cpp}
//!                               bool needs_data_on_host_function(void* data, void* parameters);
//!                            \endcode
//! can be NONE, then it is assumed this function returns false.
//! </td></tr>
//!
//! <tr><td>set_parameter_function </td><td> set the parameter, assumes the signature
//!                            \code{.cpp}
//!                                void set_parameter_function(void* parameters, const char* key, const char* value);
//!                            \endcode</td></tr>
//!
//! <tr><td>set_mpi_comm_function </td><td> set mpi communicator, assumes the signature
//!                            \code{.cpp}
//!                                void set_mpi_comm_function(void* data, void* parameters, MPI_Comm communicator);
//!                            \endcode
//!                             can be set to NONE
//!                            </td></tr>
//!
//! <tr><td>new_timestep_function </td><td> called when at the beginning of a new timestep to be written
//!                            \code{.cpp}
//!                                void new_timestep_function(void* data, void* parameters, real time, int timestep_number);
//!                            \endcode</td></tr>
//! <tr><td>end_timestep_function </td><td> called when at the end of a new timestep after all variables have been written
//!                            \code{.cpp}
//!                                void end_timestep_function(void* data, void* parameters, real time, int timestep_number);
//!                            \endcode</td></tr>
//! </table>
class DLLWriter : public Writer {
public:


    DLLWriter(const std::string& basename,
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
        const simulator::TimestepInformation& timestepInformation);


private:
    using DLLData = void*;
    DLLData dllData = nullptr;
    DLLData parametersStruct = nullptr;

    std::function<void(DLLData, DLLData, real, int)>
    newTimestepFunction;

    std::function<void(DLLData, DLLData, real, int)>
    endTimestepFunction;

    using write_function_t = void(void* /*data*/,
            void* /*parameters*/,
            real /*time*/,
            const char* /*variable_name*/,
            const real* /*variable_data*/,
            int /*nx*/,
            int /*ny*/,
            int /*nz*/,
            int /*ngx*/,
            int /*ngy*/,
            int /*ngz*/,
            real /*ax*/,
            real /*ay*/,
            real /*az*/,
            real /*bx*/,
            real /*by*/,
            real /*bz*/,
            int /*gpu_number*/ );
    std::function<write_function_t>
    writeFunction;
    std::function<void(DLLData)> deleteFunction;
    std::function<void(DLLData)> deleteParametersFunction;

    bool needsDataOnHost = false;

};
} // namespace io
} // namespace alsfvm
