#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <algorithm>
#include <math.h>

namespace py = pybind11;


double cost_function(const py::array_t<double>& x_in, const py::array_t<uint8_t>& inputimage) {
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
    // for (int k = floor(x[1] - x[2]); k < floor(x[1] + x[2]); ++k) {
    for (size_t k = y_range.first; k < y_range.second; ++ k) {
        j_pow.push_back(std::pow(k - x[1], 2));
    }
    //try {
    for (size_t i = x_range.first; i < x_range.second; ++ i) {
        // if (!(i > -1 && i < size_x - 1; ++i)) {
        //     continue;
        // }
        const auto i_squared = pow(i - x[0], 2);
        auto n = 0;
        for (size_t j = y_range.first; j < y_range.second; ++ j) {
            // if (!(j > -1 && j < size_y)) {
            //     continue;
            // }
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
    // }
    // catch (std::exception e) {
    //     py::print("Exception occurred");
    //     py::print(e.what());
    // }
    
    if (innersum == 0 || outercount == 0 || innercount == 0) {
        return 2.;
    }
    return (outersum/outercount) / (innersum/innercount);
//    return (static_cast<double>(outersum)/static_cast<double>(outercount)) / (static_cast<double>(innersum)/static_cast<double>(innercount));
}

//    size_x = len(inputimage)
//    size_y = len(inputimage[0])
//    x = [x[0]*size_x, x[1]*size_y, x[2]*max(size_x, size_y)]
//    outersum = 0
//    outercount = 0
//    innersum = 0
//    innercount = 0
//    #if x[0] < x[2] or x[0] > size_x- x[2]:
//    #    return 0, 1000
//    #if x[1] < x[2] or x[1] > size_y - x[2]:
//    #    return 0, 1000
//    j_pow = []
//    for k in range(floor(x[1] - x[2]), floor(x[1] + x[2]), 1):
//        j_pow.append(pow(k - x[1], 2))
//        
//    for i in range(floor(x[0] - x[2]), floor(x[0] + x[2]), 1):
//        if i >= size_x:
//            continue
//        i_squared = pow(i - x[0], 2)
//        n = 0
//        for j in range(floor(x[1] - x[2]), floor(x[1] + x[2]), 1):
//            if j >= size_y:
//                continue
//            local_radius = sqrt(i_squared + j_pow[n])
//            #if local_radius > x[2]:
//            #    continue
//            n +=1
//            if local_radius > x[2]:
//                outersum += pow(inputimage[i][j],2)
//                outercount +=1
//            if local_radius <=  x[2]:
//                innersum += pow(inputimage[i][j],2)
//                innercount += 1
//    #if outercount == 0 or innercount == 0:
//    #    return 0, 1000
//        
//    return innersum/innercount, outersum/outercount
namespace py = pybind11;

PYBIND11_MODULE(find_coin, m) {
  m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: python_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
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

