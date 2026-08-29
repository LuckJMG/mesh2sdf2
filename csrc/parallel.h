#ifndef PARALLEL_H
#define PARALLEL_H

#include "array3.h"
#include "geometry.h"

#include <stddef.h>

static constexpr size_t sweep_parallel_min_cells = 1 << 18;
static constexpr size_t init_parallel_min_tris = 1 << 12;

void sweep_parallel(const TriangleData *triangle_data, Array3f &phi,
					Array3i &closest_tri, const Vec3f &origin, float dx, int di,
					int dj, int dk);

void init_distances_and_counts_parallel(const Vec3ui *tri, int ntri,
										const Vec3f *x, const Vec3f &origin,
										float dx, int ni, int nj, int nk,
										const int exact_band, Array3f &phi,
										Array3i &closest_tri,
										Array3i &intersection_count,
										TriangleData *triangle_data);

#endif
