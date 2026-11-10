#include "makelevelset3.h"

#include "geometry.h"
#include "parallel.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static void check_neighbour(const TriangleData *triangle_data, Array3f &phi,
							Array3i &closest_tri, const Vec3f &gx, int i0,
							int j0, int k0, int i1, int j1, int k1) {
	int t = closest_tri(i1, j1, k1);
	if (t >= 0 && t != closest_tri(i0, j0, k0)) {
		float d = point_triangle_distance(gx, triangle_data[t]);
		if (d < phi(i0, j0, k0)) {
			phi(i0, j0, k0) = d;
			closest_tri(i0, j0, k0) = t;
		}
	}
}

static void update_cell(const TriangleData *triangle_data, Array3f &phi,
						Array3i &closest_tri, const Grid &grid, int i, int j,
						int k, int di, int dj, int dk) {
	Vec3f gx((float)i * grid.dx + grid.origin.x,
			 (float)j * grid.dx + grid.origin.y,
			 (float)k * grid.dx + grid.origin.z);

	check_neighbour(triangle_data, phi, closest_tri, gx, i, j, k, i - di, j, k);
	check_neighbour(triangle_data, phi, closest_tri, gx, i, j, k, i, j - dj, k);
	check_neighbour(triangle_data, phi, closest_tri, gx, i, j, k, i - di,
					j - dj, k);
	check_neighbour(triangle_data, phi, closest_tri, gx, i, j, k, i, j, k - dk);
	check_neighbour(triangle_data, phi, closest_tri, gx, i, j, k, i - di, j,
					k - dk);
	check_neighbour(triangle_data, phi, closest_tri, gx, i, j, k, i, j - dj,
					k - dk);
	check_neighbour(triangle_data, phi, closest_tri, gx, i, j, k, i - di,
					j - dj, k - dk);
}

static void sweep(const TriangleData *triangle_data, Array3f &phi,
				  Array3i &closest_tri, const Grid &grid, int di, int dj,
				  int dk) {
	int i0 = di > 0 ? 1 : phi.ni - 2;
	int i1 = di > 0 ? phi.ni : -1;

	int j0 = dj > 0 ? 1 : phi.nj - 2;
	int j1 = dj > 0 ? phi.nj : -1;

	int k0 = dk > 0 ? 1 : phi.nk - 2;
	int k1 = dk > 0 ? phi.nk : -1;

	for (int k = k0; k != k1; k += dk) {
		for (int j = j0; j != j1; j += dj) {
			for (int i = i0; i != i1; i += di) {
				update_cell(triangle_data, phi, closest_tri, grid, i, j, k, di,
							dj, dk);
			}
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
static void
apply_signs_from_intersection_counts(const Array3i &intersection_count,
									 Array3f &phi, int ni, int nj, int nk) {
#pragma omp parallel for collapse(2)
	for (int k = 0; k < nk; ++k) {
		for (int j = 0; j < nj; ++j) {
			int total_count = 0;
			for (int i = 0; i < ni; ++i) {
				total_count += intersection_count(i, j, k);
				if (total_count % 2 ==
					1) { // if parity of intersections so far is odd,
					phi(i, j, k) = -phi(i, j, k); // we are inside the mesh
				}
			}
		}
	}
}

// fill in the rest of the distances with fast sweeping
static void run_fast_sweeping_passes(const TriangleData *triangle_data,
									 Array3f &phi, Array3i &closest_tri,
									 const Grid &grid) {
	void (*sweep_once)(const TriangleData *, Array3f &, Array3i &, const Grid &,
					   int, int, int) = sweep;
#ifdef _OPENMP
	if (omp_get_max_threads() > 1 &&
		(size_t)phi.ni * phi.nj * phi.nk >= sweep_parallel_min_cells) {
		sweep_once = sweep_parallel;
	}
#endif
	for (unsigned int pass = 0; pass < 2; ++pass) {
		sweep_once(triangle_data, phi, closest_tri, grid, +1, +1, +1);
		sweep_once(triangle_data, phi, closest_tri, grid, -1, -1, -1);
		sweep_once(triangle_data, phi, closest_tri, grid, +1, +1, -1);
		sweep_once(triangle_data, phi, closest_tri, grid, -1, -1, +1);
		sweep_once(triangle_data, phi, closest_tri, grid, +1, -1, +1);
		sweep_once(triangle_data, phi, closest_tri, grid, -1, +1, -1);
		sweep_once(triangle_data, phi, closest_tri, grid, +1, -1, -1);
		sweep_once(triangle_data, phi, closest_tri, grid, -1, +1, +1);
	}
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

	// init_distances_and_counts() fills phi and closest_tri.
	Array3i closest_tri;
	closest_tri.resize(ni, nj, nk);
	Array3i intersection_count(ni, nj, nk,
							   0); // intersection_count(i,j,k) is # of tri
								   // intersections in (i-1,i]x{j}x{k}
	TriangleData *triangle_data =
		(TriangleData *)calloc((size_t)ntri, sizeof(TriangleData));
	assert(triangle_data != NULL);

	Grid grid = {origin, dx, ni, nj, nk};
	init_distances_and_counts(tri, ntri, vertices, grid, phi, closest_tri,
							  intersection_count, triangle_data);

	assert(phi.storage != NULL);
	assert(closest_tri.storage != NULL);
	assert(intersection_count.storage != NULL);

	run_fast_sweeping_passes(triangle_data, phi, closest_tri, grid);
	apply_signs_from_intersection_counts(intersection_count, phi, ni, nj, nk);
	free(triangle_data);
}
