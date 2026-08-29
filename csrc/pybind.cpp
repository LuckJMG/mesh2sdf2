#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string.h>
#include <vector>

#include "makelevelset3.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

py::array_t<float> compute(const py::array_t<float> &vertices,
						   const py::array_t<unsigned int> &faces, int size) {
	std::vector<Vec3f> V;
	V.reserve(vertices.shape(0));
	for (int i = 0; i < vertices.shape(0); ++i) {
		V.push_back(
			Vec3f(vertices.at(i, 0), vertices.at(i, 1), vertices.at(i, 2)));
	}

	std::vector<Vec3ui> F;
	F.reserve(faces.shape(0));
	for (int i = 0; i < faces.shape(0); ++i) {
		F.push_back(Vec3ui(faces.at(i, 0), faces.at(i, 1), faces.at(i, 2)));
	}

	Vec3f bbmin(-1.0f, -1.0f, -1.0f);
	Vec3f bbmax(1.0f, 1.0f, 1.0f);
	if (size < 2) {
		throw py::value_error("size must be at least 2");
	}

	float dx = 2.0f / (float)(size - 1);
	Array3f grid;
	{
		py::gil_scoped_release release;
		make_level_set3(F, V, bbmin, dx, size, size, size, grid);
	}

	py::array_t<float> sdf({size, size, size});
	memcpy(sdf.mutable_data(), grid.storage,
		   (size_t)size * size * size * sizeof(float));
	return sdf;
}

PYBIND11_MODULE(core, m) {
	m.def("compute", &compute, R"pbdoc(
        Compute the SDF from an input mesh.

        Args:
          vertices (np.ndarray): The vertex array with shape (Nv, 3), and
              vertices MUST be in range [-1, 1].
          faces (np.ndarray): The face array with shape (Nf, 3).
          size (int): The resolution of resulting SDF.
        )pbdoc",
		  py::arg("vertices"), py::arg("faces"), py::arg("size") = 128);

#ifdef VERSION_INFO
	m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
	m.attr("__version__") = "dev";
#endif
}
