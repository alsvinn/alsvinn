#include <fbm/fbm.hpp>
#include <fbmpy/fbmpy.hpp>
namespace fbmpy {
//! Adapter class for adding methods into boost::python
//!
//! Yes, this looks a bit stupid, but it was the only way I found to
//! dynamically add funtions to the python environment
struct FBMPY {
    static boost::python::numpy::ndarray fractional_brownian_motion_1d(double H,
        int nx,
        const boost::python::numpy::ndarray& X) {
        return fbmpy::fractional_brownian_motion_1d(H, nx, X);
    }

    static boost::python::numpy::ndarray fractional_brownian_bridge_1d(double H,
        int nx,
        const boost::python::numpy::ndarray& X) {
        return fbmpy::fractional_brownian_bridge_1d(H, nx, X);
    }


    static boost::python::numpy::ndarray fractional_brownian_motion_2d(double H,
        int nx,
        const boost::python::numpy::ndarray& X) {
        return fbmpy::fractional_brownian_motion_2d(H, nx, X);
    }

    static boost::python::numpy::ndarray fractional_brownian_bridge_2d(double H,
        int nx,
        const boost::python::numpy::ndarray& X) {
        return fbmpy::fractional_brownian_bridge_2d(H, nx, X);
    }

    static boost::python::numpy::ndarray fractional_brownian_motion_3d(double H,
        int nx,
        const boost::python::numpy::ndarray& X) {
        return fbmpy::fractional_brownian_motion_3d(H, nx, X);
    }

    static boost::python::numpy::ndarray fractional_brownian_bridge_3d(double H,
        int nx,
        const boost::python::numpy::ndarray& X) {
        return fbmpy::fractional_brownian_bridge_3d(H, nx, X);
    }
};
inline void addFBMToPython(boost::python::object& module) {
    //1d
    module.attr("fbmpy") = boost::python::class_<FBMPY>("fbmpy")
        .def("fractional_brownian_motion_1d", FBMPY::fractional_brownian_motion_1d)
        .staticmethod("fractional_brownian_motion_1d")
        .def("fractional_brownian_bridge_1d", FBMPY::fractional_brownian_bridge_1d)
        .staticmethod("fractional_brownian_bridge_1d")

            //2d
        .def("fractional_brownian_motion_2d", FBMPY::fractional_brownian_motion_2d)
        .staticmethod("fractional_brownian_motion_2d")
        .def("fractional_brownian_bridge_2d", FBMPY::fractional_brownian_bridge_2d)
        .staticmethod("fractional_brownian_bridge_2d")

            //3d
        .def("fractional_brownian_motion_3d", FBMPY::fractional_brownian_motion_3d)
        .staticmethod("fractional_brownian_motion_3d")
        .def("fractional_brownian_bridge_3d", FBMPY::fractional_brownian_bridge_3d)
        .staticmethod("fractional_brownian_bridge_3d");
}

}
