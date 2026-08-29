#ifndef MAKELEVELSET3_H
#define MAKELEVELSET3_H

#include "array3.h"
#include "vec.h"

#include <vector>

// Distances exact within exact_band cells; farther cells use nearby triangle.
// Modern C++ bridge — keeps std::vector for Python bindings.
void make_level_set3(const std::vector<Vec3ui> &tri,
					 const std::vector<Vec3f> &x, const Vec3f &origin, float dx,
					 int nx, int ny, int nz, Array3f &phi,
					 const int exact_band = 1);

// Orthodox core — raw pointers, no STL allocation.
void make_level_set3(const Vec3ui *tri, int ntri, const Vec3f *x,
					 const Vec3f &origin, float dx, int nx, int ny, int nz,
					 Array3f &phi, const int exact_band = 1);

#endif
