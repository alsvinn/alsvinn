#pragma once
#include <mpi.h>
#include "dll_writer_example_exports.h"
#ifdef __cplusplus
extern "C" {
#endif


DLL_WRITER_EXPORT void* create(const char* simulator_name,
    const char* simulator_version, void* parameters);

DLL_WRITER_EXPORT void delete_data(void* data);

DLL_WRITER_EXPORT void write(void* data, void* parameters, double time,
    const char* variable_name, const double* variable_data, int nx, int ny, int nz,
    int ngx, int ngy, int ngz, double ax, double ay, double az, double bx,
    double by, double bz, int gpu_number );

DLL_WRITER_EXPORT void* make_parameters();

DLL_WRITER_EXPORT void delete_parameters(void* parameters);

DLL_WRITER_EXPORT bool needs_data_on_host(void* data, void* parameters);

DLL_WRITER_EXPORT void set_parameter(void* parameters, const char* key,
    const char* value);

DLL_WRITER_EXPORT void set_mpi_comm(void* data, void* parameters,
    MPI_Comm communicator);

DLL_WRITER_EXPORT void new_timestep(void* data, void* parameters, double time,
    int timestep_number);

DLL_WRITER_EXPORT void end_timestep(void* data, void* parameters, double time,
    int timestep_number);
#ifdef __cplusplus
}
#endif
