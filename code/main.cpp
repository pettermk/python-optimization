#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;


int cost_function(py::array_t<double> x, py::array_t<int> inputimage) {
    py::print(x.size());
    auto r = inputimage.unchecked<2>();
    for (ssize_t i = 0; i < r.shape(0); i++) {
        for (ssize_t j = 0; j < r.shape(1); j++) {
            py::print(r(i,j));
        }
    }
    return 50;
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

