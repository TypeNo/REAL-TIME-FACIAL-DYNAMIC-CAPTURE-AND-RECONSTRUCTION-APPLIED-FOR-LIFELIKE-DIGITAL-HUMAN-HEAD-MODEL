#include <pybind11/pybind11.h>
#include "progress_shared.hpp"
#include <pybind11/embed.h>

namespace py = pybind11;

/*PYBIND11_MODULE(mybridge, m) {
    m.def("update_progress", &update_progress);
    m.def("get_current_progress",&get_current_progress);
    m.def("get_total_progress",&get_total_progress);
}*/

PYBIND11_EMBEDDED_MODULE(mybridge, m) {
    m.def("update_progress", &update_progress);
    m.def("get_current_progress",&get_current_progress);
    m.def("get_total_progress",&get_total_progress);
    m.def("update_model_path",&update_model_path);
    m.def("get_model_path",&get_model_path);
    m.def("update_FPS",&update_FPS);
    m.def("get_FPS",&get_FPS);
    m.def("update_expression_path",&update_expression_path);
    m.def("get_expression_path",&get_expression_path);

}