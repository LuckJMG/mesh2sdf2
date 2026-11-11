#ifndef MAKELEVELSET3_H
#define MAKELEVELSET3_H

#include "array3.h"
#include "geometry.h"
#include "vec.h"

// Distances are exact within exact_band cells; farther cells use a nearby
// triangle.
void make_level_set3(const Vec3ui *tri, int ntri, const Vec3f *vertices,
					 const Vec3f &origin, float dx, int nx, int ny, int nz,
					 Array3f &phi);

// The two halves of make_level_set3() around the sweeps, for a sweep on
// another device. init_level_set3() fills every cell of the three arrays and
// one TriangleData per triangle.
void init_level_set3(const Vec3ui *tri, int ntri, const Vec3f *vertices,
					 const Grid &grid, Array3f &phi, Array3i &closest_tri,
					 Array3i &intersection_count, TriangleData *triangle_data);
void apply_signs(const int *intersection_count, float *phi, int ni, int nj,
				 int nk);

#endif
