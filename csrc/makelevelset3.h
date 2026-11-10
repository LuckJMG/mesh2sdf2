#ifndef MAKELEVELSET3_H
#define MAKELEVELSET3_H

#include "array3.h"
#include "vec.h"

// Distances are exact within exact_band cells; farther cells use a nearby
// triangle.
void make_level_set3(const Vec3ui *tri, int ntri, const Vec3f *x,
					 const Vec3f &origin, float dx, int nx, int ny, int nz,
					 Array3f &phi);

#endif
