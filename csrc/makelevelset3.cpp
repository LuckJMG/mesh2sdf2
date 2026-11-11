#include "makelevelset3.h"

#include "geometry.h"
#include "parallel.h"

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static bool check_neighbour(const TriangleData *triangle_data, Array3f &phi,
							Array3i &closest_tri, const Vec3f &gx, int i0,
							int j0, int k0, int i1, int j1, int k1) {
	int t = closest_tri(i1, j1, k1);
	if (t >= 0 && t != closest_tri(i0, j0, k0)) {
		float d = point_triangle_distance(gx, triangle_data[t]);
		if (d < phi(i0, j0, k0)) {
			phi(i0, j0, k0) = d;
			closest_tri(i0, j0, k0) = t;
			return true;
		}
	}
	return false;
}

static bool update_cell(const TriangleData *triangle_data, Array3f &phi,
						Array3i &closest_tri, const Grid &grid, int i, int j,
						int k, int di, int dj, int dk) {
	Vec3f gx((float)i * grid.dx + grid.origin.x,
			 (float)j * grid.dx + grid.origin.y,
			 (float)k * grid.dx + grid.origin.z);

	bool changed = check_neighbour(triangle_data, phi, closest_tri, gx, i, j, k,
								   i - di, j, k);
	changed |= check_neighbour(triangle_data, phi, closest_tri, gx, i, j, k, i,
							   j - dj, k);
	changed |= check_neighbour(triangle_data, phi, closest_tri, gx, i, j, k,
							   i - di, j - dj, k);
	changed |= check_neighbour(triangle_data, phi, closest_tri, gx, i, j, k, i,
							   j, k - dk);
	changed |= check_neighbour(triangle_data, phi, closest_tri, gx, i, j, k,
							   i - di, j, k - dk);
	changed |= check_neighbour(triangle_data, phi, closest_tri, gx, i, j, k, i,
							   j - dj, k - dk);
	changed |= check_neighbour(triangle_data, phi, closest_tri, gx, i, j, k,
							   i - di, j - dj, k - dk);
	return changed;
}

static void sweep_row(const TriangleData *triangle_data, Array3f &phi,
					  Array3i &closest_tri, const Grid &grid, int j, int k,
					  int di, int dj, int dk, const RowSkip &rows) {
	int i0 = di > 0 ? 1 : phi.ni - 2;
	int i1 = di > 0 ? phi.ni : -1;

	bool changed = false;
	if (row_unchanged(rows, phi.nj, phi.nk, j, k, dj, dk)) {
		// the last cell lies on the i boundary, which misses some sweeps
		changed = update_cell(triangle_data, phi, closest_tri, grid, i1 - di, j,
							  k, di, dj, dk);
	}
	else {
		for (int i = i0; i != i1; i += di) {
			changed |= update_cell(triangle_data, phi, closest_tri, grid, i, j,
								   k, di, dj, dk);
		}
	}
	if (changed) {
		rows.last_change[j + phi.nj * k] = (signed char)rows.sweep;
	}
}

static void sweep(const TriangleData *triangle_data, Array3f &phi,
				  Array3i &closest_tri, const Grid &grid, int di, int dj,
				  int dk, const RowSkip &rows) {
	int j0 = dj > 0 ? 1 : phi.nj - 2;
	int j1 = dj > 0 ? phi.nj : -1;

	int k0 = dk > 0 ? 1 : phi.nk - 2;
	int k1 = dk > 0 ? phi.nk : -1;

	for (int k = k0; k != k1; k += dk) {
		for (int j = j0; j != j1; j += dj) {
			sweep_row(triangle_data, phi, closest_tri, grid, j, k, di, dj, dk,
					  rows);
		}
	}
}

// initialize distances near the mesh within exact_band cells of each triangle,
// and record triangle intersections along each grid row for later sign
// determination
static void init_distances_and_counts(const Vec3ui *tri, int ntri,
									  const Vec3f *vertices, const Grid &grid,
									  Array3f &phi, Array3i &closest_tri,
									  Array3i &intersection_count,
									  TriangleData *triangle_data) {
#ifdef _OPENMP
	if (omp_get_max_threads() > 1 && (size_t)ntri >= init_parallel_min_tris) {
		init_distances_and_counts_parallel(tri, ntri, vertices, grid, phi,
										   closest_tri, intersection_count,
										   triangle_data);
		return;
	}
#endif
	const int ni = grid.ni, nj = grid.nj, nk = grid.nk;
	const float dx = grid.dx;
	const Vec3f &origin = grid.origin;
	phi.assign((float)(ni + nj + nk) * dx); // upper bound on distance
	closest_tri.assign(-1);
	for (int t = 0; t < ntri; ++t) {
		TriSetup setup = setup_triangle(tri[t], vertices, origin, dx);
		triangle_data[t] = setup.triangle;

		IndexRange band_i = band_range(setup.p_i, setup.q_i, setup.r_i, ni);
		IndexRange band_j = band_range(setup.p_j, setup.q_j, setup.r_j, nj);
		IndexRange band_k = band_range(setup.p_k, setup.q_k, setup.r_k, nk);
		for (int k = band_k.lo; k <= band_k.hi; ++k) {
			for (int j = band_j.lo; j <= band_j.hi; ++j) {
				for (int i = band_i.lo; i <= band_i.hi; ++i) {
					Vec3f gx((float)i * dx + origin.x, (float)j * dx + origin.y,
							 (float)k * dx + origin.z);
					float d = point_triangle_distance(gx, setup.triangle);
					if (d < phi(i, j, k)) {
						phi(i, j, k) = d;
						closest_tri(i, j, k) = t;
					}
				}
			}
		}

		IndexRange rows_j = row_range(setup.p_j, setup.q_j, setup.r_j, nj);
		IndexRange rows_k = row_range(setup.p_k, setup.q_k, setup.r_k, nk);
		for (int k = rows_k.lo; k <= rows_k.hi; ++k) {
			for (int j = rows_j.lo; j <= rows_j.hi; ++j) {
				double a, b, c;
				if (point_in_triangle_2d(j, k, setup.p_j, setup.p_k, setup.q_j,
										 setup.q_k, setup.r_j, setup.r_k, a, b,
										 c)) {
					double fi = a * setup.p_i + b * setup.q_i + c * setup.r_i;
					int i_interval = (int)ceil(fi);
					if (i_interval < 0) {
						++intersection_count(0, j, k);
					}
					else if (i_interval < ni) {
						++intersection_count(i_interval, j, k);
					}
				}
			}
		}
	}
}

// figure out signs (inside/outside) from intersection counts
// Kept adjacent to init_distances_and_counts: both handle parity via
// intersection_count, while the sweep phase is distance-only.
void apply_signs(const int *intersection_count, float *phi, int ni, int nj,
				 int nk) {
#pragma omp parallel for collapse(2)
	for (int k = 0; k < nk; ++k) {
		for (int j = 0; j < nj; ++j) {
			size_t row = (size_t)ni * ((size_t)j + (size_t)nj * k);
			int total_count = 0;
			for (int i = 0; i < ni; ++i) {
				total_count += intersection_count[row + i];
				if (total_count % 2 ==
					1) { // if parity of intersections so far is odd,
					phi[row + i] = -phi[row + i]; // we are inside the mesh
				}
			}
		}
	}
}

void init_level_set3(const Vec3ui *tri, int ntri, const Vec3f *vertices,
					 const Grid &grid, Array3f &phi, Array3i &closest_tri,
					 Array3i &intersection_count, TriangleData *triangle_data) {
	phi.resize(grid.ni, grid.nj, grid.nk);
	closest_tri.resize(grid.ni, grid.nj, grid.nk);
	// intersection_count(i,j,k) is # of tri intersections in (i-1,i]x{j}x{k}
	intersection_count.resize(grid.ni, grid.nj, grid.nk);
	intersection_count.assign(0);
	init_distances_and_counts(tri, ntri, vertices, grid, phi, closest_tri,
							  intersection_count, triangle_data);
}

static const int sweep_dirs[8][3] = {{+1, +1, +1}, {-1, -1, -1}, {+1, +1, -1},
									 {-1, -1, +1}, {+1, -1, +1}, {-1, +1, -1},
									 {+1, -1, -1}, {-1, +1, +1}};
static const int sweep_count = 16; // two passes over the 8 directions

// An interior cell checks offset o in every sweep whose direction matches the
// nonzero components of o. If the source row did not change since the last
// such sweep, the check repeats a triangle that already lost, and phi only
// decreases, so the check cannot change the cell. Skipping it keeps the
// result bit-identical.
static void set_row_limits(int s, int limit[4]) {
	const int *d = sweep_dirs[s % 8];
	for (int r = 0; r < 4; ++r) {
		limit[r] = SCHAR_MAX;
	}
	for (int o = 1; o < 8; ++o) {
		int a = o & 1, b = (o >> 1) & 1, c = (o >> 2) & 1;
		int last = -2;
		for (int q = s - 1; q >= 0 && last < 0; --q) {
			const int *e = sweep_dirs[q % 8];
			if ((!a || e[0] == d[0]) && (!b || e[1] == d[1]) &&
				(!c || e[2] == d[2])) {
				last = q;
			}
		}
		if (last < limit[b + 2 * c]) {
			limit[b + 2 * c] = last;
		}
	}
}

// fill in the rest of the distances with fast sweeping
static void run_fast_sweeping_passes(const TriangleData *triangle_data,
									 Array3f &phi, Array3i &closest_tri,
									 const Grid &grid) {
	void (*sweep_once)(const TriangleData *, Array3f &, Array3i &, const Grid &,
					   int, int, int, const RowSkip &) = sweep;
#ifdef _OPENMP
	if (omp_get_max_threads() > 1 &&
		(size_t)phi.ni * phi.nj * phi.nk >= sweep_parallel_min_cells) {
		sweep_once = sweep_parallel;
	}
#endif
	size_t nrows = (size_t)phi.nj * phi.nk;
	RowSkip rows;
	rows.last_change = (signed char *)malloc(nrows);
	assert(rows.last_change != NULL);
	memset(rows.last_change, -1, nrows);
	for (rows.sweep = 0; rows.sweep < sweep_count; ++rows.sweep) {
		const int *d = sweep_dirs[rows.sweep % 8];
		set_row_limits(rows.sweep, rows.limit);
		sweep_once(triangle_data, phi, closest_tri, grid, d[0], d[1], d[2],
				   rows);
	}
	free(rows.last_change);
}

void make_level_set3(const Vec3ui *tri, int ntri, const Vec3f *vertices,
					 const Vec3f &origin, float dx, int ni, int nj, int nk,
					 Array3f &phi) {
	assert(ni > 0 && nj > 0 && nk > 0);
	if (ni <= 0 || nj <= 0 || nk <= 0) {
		return;
	}

	phi.resize(ni, nj, nk);
	if (ntri == 0) {
		phi.assign((float)(ni + nj + nk) * dx);
		return;
	}

	Array3i closest_tri;
	Array3i intersection_count;
	TriangleData *triangle_data =
		(TriangleData *)calloc((size_t)ntri, sizeof(TriangleData));
	assert(triangle_data != NULL);

	Grid grid = {origin, dx, ni, nj, nk};
	init_level_set3(tri, ntri, vertices, grid, phi, closest_tri,
					intersection_count, triangle_data);

	assert(phi.storage != NULL);
	assert(closest_tri.storage != NULL);
	assert(intersection_count.storage != NULL);

	run_fast_sweeping_passes(triangle_data, phi, closest_tri, grid);
	apply_signs(intersection_count.storage, phi.storage, ni, nj, nk);
	free(triangle_data);
}
