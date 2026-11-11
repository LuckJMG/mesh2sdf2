#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <string.h>
#include <vector>

#include "makelevelset3.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

static const double BOX_MIN = -1.0;

static double grid_dx(int size) {
	if (size < 2) {
		throw py::value_error("size must be at least 2");
	}
	return 2.0 / (size - 1);
}

struct Mesh {
	std::vector<Vec3f> vertices;
	std::vector<Vec3ui> faces;
};

static Mesh read_mesh(const py::array_t<float> &vertices,
					  const py::array_t<unsigned int> &faces) {
	Mesh mesh;
	mesh.vertices.reserve(vertices.shape(0));
	for (int i = 0; i < vertices.shape(0); ++i) {
		mesh.vertices.push_back(
			Vec3f(vertices.at(i, 0), vertices.at(i, 1), vertices.at(i, 2)));
	}
	mesh.faces.reserve(faces.shape(0));
	for (int i = 0; i < faces.shape(0); ++i) {
		mesh.faces.push_back(
			Vec3ui(faces.at(i, 0), faces.at(i, 1), faces.at(i, 2)));
	}
	return mesh;
}

py::array_t<float> compute(const py::array_t<float> &vertices,
						   const py::array_t<unsigned int> &faces, int size) {
	float dx = (float)grid_dx(size);
	Mesh mesh = read_mesh(vertices, faces);

	Vec3f bbmin((float)BOX_MIN, (float)BOX_MIN, (float)BOX_MIN);
	Array3f grid;
	{
		py::gil_scoped_release release;
		make_level_set3(mesh.faces.data(), (int)mesh.faces.size(),
						mesh.vertices.data(), bbmin, dx, size, size, size,
						grid);
	}

	py::array_t<float> sdf({size, size, size});
	memcpy(sdf.mutable_data(), grid.storage,
		   (size_t)size * size * size * sizeof(float));
	return sdf;
}

// Leaves `a` empty. The NumPy array owns and frees the storage.
template <class T> static py::array_t<T> take_array(Array3<T> &a) {
	T *data = a.storage;
	a.storage = nullptr;
	py::capsule owner(data, [](void *p) { free(p); });
	return py::array_t<T>({a.nk, a.nj, a.ni}, data, owner);
}

static py::tuple init(const py::array_t<float> &vertices,
					  const py::array_t<unsigned int> &faces, int size) {
	float dx = (float)grid_dx(size);
	Mesh mesh = read_mesh(vertices, faces);
	int ntri = (int)mesh.faces.size();

	Vec3f bbmin((float)BOX_MIN, (float)BOX_MIN, (float)BOX_MIN);
	Grid grid = {bbmin, dx, size, size, size};
	Array3f phi;
	Array3i closest_tri, intersection_count;
	std::vector<TriangleData> triangle_data(ntri);
	{
		py::gil_scoped_release release;
		init_level_set3(mesh.faces.data(), ntri, mesh.vertices.data(), grid,
						phi, closest_tri, intersection_count,
						triangle_data.data());
	}

	// float32 rows in the order of TriangleData in mesh2sdf/sweep.cl. Each
	// squared_length is a float widened to double, so float32 holds it exactly.
	py::array_t<float> triangles({ntri, 31});
	float *row = triangles.mutable_data();
	for (const TriangleData &t : triangle_data) {
		for (const Vec3f &v : {t.x1, t.x2, t.x3, t.x13, t.x23}) {
			*row++ = v.x;
			*row++ = v.y;
			*row++ = v.z;
		}
		for (float f : {t.m13, t.m23, t.dot, t.invdet}) {
			*row++ = f;
		}
		for (const SegmentData &e : {t.edge12, t.edge13, t.edge23}) {
			*row++ = e.delta.x;
			*row++ = e.delta.y;
			*row++ = e.delta.z;
		}
		for (const SegmentData &e : {t.edge12, t.edge13, t.edge23}) {
			*row++ = (float)e.squared_length;
		}
	}
	return py::make_tuple(take_array(phi), take_array(closest_tri),
						  take_array(intersection_count), triangles);
}

static void
apply_signs_numpy(py::array_t<float, py::array::c_style> &phi,
				  const py::array_t<int, py::array::c_style> &count) {
	if (phi.ndim() != 3 || count.ndim() != 3 ||
		!std::equal(phi.shape(), phi.shape() + 3, count.shape())) {
		throw py::value_error("phi and intersection_count must have the same "
							  "3D shape");
	}
	float *p = phi.mutable_data();
	const int *c = count.data();
	int nk = (int)phi.shape(0), nj = (int)phi.shape(1), ni = (int)phi.shape(2);
	py::gil_scoped_release release;
	apply_signs(c, p, ni, nj, nk);
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
	m.def("init", &init, R"pbdoc(
        Run the CPU init of `compute` and return its state for a sweep on
        another device: (phi, closest_tri, intersection_count, triangles).
        The first three have shape (size, size, size). `triangles` has one
        float32 row of 31 values per face.
        )pbdoc",
		  py::arg("vertices"), py::arg("faces"), py::arg("size") = 128);
	m.def("apply_signs", &apply_signs_numpy,
		  "Negate `phi` in place inside the mesh, from the intersection counts "
		  "of `init`.",
		  py::arg("phi").noconvert(), py::arg("intersection_count"));
	m.def("grid_dx", &grid_dx, "Grid spacing for `size` samples over the box.",
		  py::arg("size"));
	m.attr("box_min") = BOX_MIN;

#ifdef VERSION_INFO
	m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
	m.attr("__version__") = "dev";
#endif
}
