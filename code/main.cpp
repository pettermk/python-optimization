#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <algorithm>
#include <math.h>

namespace py = pybind11;


double cost_function(const py::array_t<double> x_in, const py::array_t<uint8_t> inputimage) {
    const auto x_raw = x_in.unchecked<1>();
    const auto r = inputimage.unchecked<2>();
    const double size_x = r.shape(0);
    const double size_y = r.shape(1);
    auto outersum = 0.;
    auto outercount = 0.;
    auto innersum = 0.;
    auto innercount = 0.;
    auto j_pow = std::vector<long long>();
    const auto x = std::vector<double>{x_raw(0)*size_x, x_raw(1)*size_y, x_raw(2)*std::max(size_x, size_y)};
    const std::pair<size_t, size_t> x_range = {static_cast<size_t>(std::max(0., std::floor(x[0] - x[2]))),
                                         static_cast<size_t>(std::min(double(size_x), std::floor(x[0] + x[2])))};
    const std::pair<size_t, size_t> y_range = {static_cast<size_t>(std::max(0., std::floor(x[1] - x[2]))),
                                         static_cast<size_t>(std::min(double(size_y), std::floor(x[1] + x[2])))};
    for (size_t k = y_range.first; k < y_range.second; ++ k) {
        j_pow.push_back(std::pow(k - x[1], 2));
    }
    for (size_t i = x_range.first; i < x_range.second; ++ i) {
        const auto i_squared = pow(i - x[0], 2);
        auto n = 0;
        for (size_t j = y_range.first; j < y_range.second; ++ j) {
            const double local_radius = sqrt(i_squared + j_pow[n]);
            ++n;
            if (local_radius > x[2]) {
                outersum += std::pow(r(i, j), 2);
                ++outercount;
            }
            else if (local_radius <= x[2]) {
                innersum += std::pow(r(i, j), 2);
                ++innercount;
            }
        }
    }
    
    if (innersum == 0 || outercount == 0 || innercount == 0) {
        return 2.;
    }
    return (outersum/outercount) / (innersum/innercount);
}


namespace py = pybind11;

PYBIND11_MODULE(find_coin, m) {
  m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: python_example
        .. autosummary::
           :toctree: _generate
           cost_function
    )pbdoc";

  m.def("cost_function", &cost_function, R"pbdoc(
      Estimate the likelihood of a coin being present at the given coordinate
    )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}

