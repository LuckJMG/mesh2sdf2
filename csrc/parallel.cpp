#include "parallel.h"

#include "geometry.h"

#include <assert.h>
#include <atomic>
#include <math.h>
#include <stdint.h>
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

// Wavefront-parallel variant of sweep(). Rows along i are grouped by
// w = dj*j + dk*k. Each row runs serially in direction di, as in sweep(). Every
// source read by a cell lies earlier in its own row or on a row 1-2 levels
// behind it, and rows on one level never touch each other, so each level runs
// as one omp parallel for. This is a topological order of the same dependency
// DAG as sweep(), so results are bit-identical at any thread count. The same
// holds for rows.last_change: a row writes only its own entry, and reads the
// entries of rows on earlier levels.
void sweep_parallel(const TriangleData *triangle_data, Array3f &phi,
					Array3i &closest_tri, const Grid &grid, int di, int dj,
					int dk, const RowSkip &rows) {
	const int j_lo = dj > 0 ? 1 : 0;
	const int j_hi = dj > 0 ? phi.nj - 1 : phi.nj - 2;

	const int k_lo = dk > 0 ? 1 : 0;
	const int k_hi = dk > 0 ? phi.nk - 1 : phi.nk - 2;

	int w_min =
		(dj > 0 ? dj * j_lo : dj * j_hi) + (dk > 0 ? dk * k_lo : dk * k_hi);
	int w_max =
		(dj > 0 ? dj * j_hi : dj * j_lo) + (dk > 0 ? dk * k_hi : dk * k_lo);
	for (int w = w_min; w <= w_max; ++w) {
		// only rows with a valid k, so schedule(static) splits real work
		int ja = dj * (w - dk * k_lo), jb = dj * (w - dk * k_hi);
		int jmin = clamp_int(ja < jb ? ja : jb, j_lo, j_hi + 1);
		int jmax = clamp_int(ja < jb ? jb : ja, j_lo - 1, j_hi);
#pragma omp parallel for schedule(static)
		for (int j = jmin; j <= jmax; ++j) {
			int k = dk * (w - dj * j);
			sweep_row(triangle_data, phi, closest_tri, grid, j, k, di, dj, dk,
					  rows);
		}
	}
}

// Minimum-update of one cell in an array of words packed as
// (distance bits << 32) | triangle index. Distances written during init are
// non-negative, and IEEE non-negative floats order by their bit pattern, so
// integer comparison on the word is a minimum on distance with ties broken
// by the smaller triangle index. The serial loop updates on strict < while
// scanning triangles in ascending order, so this reproduces its winner at
// every cell bitwise.
static void packed_min_distance(std::atomic<uint64_t> &cell, float d,
								unsigned int t) {
	uint32_t bits;
	static_assert(sizeof(bits) == sizeof(d), "32-bit float required");
	memcpy(&bits, &d, sizeof(bits));

	uint64_t packed = ((uint64_t)bits << 32) | t;
	uint64_t cur = cell.load(std::memory_order_relaxed);
	while (packed < cur) {
		if (cell.compare_exchange_weak(cur, packed)) {
			break;
		}
	}
}

// Parallel variant of init_distances_and_counts() for dense meshes. Triangle
// bands overlap rarely once a mesh has enough triangles to reach this path,
// so a CAS minimum-update beats thread-private buffers merged at the end: no
// merge pass and O(cells) scratch instead of O(threads * cells). The initial
// upper bound (ni+nj+nk)*dx exceeds every possible distance inside the box
// (diagonal 2*sqrt(3)), so untouched cells keep the sentinel: phi_init bits
// and closest_tri -1. The unpack pass writes every cell, so the caller does
// not fill phi or closest_tri first.
void init_distances_and_counts_parallel(const Vec3ui *tri, int ntri,
										const Vec3f *vertices, const Grid &grid,
										Array3f &phi, Array3i &closest_tri,
										Array3i &intersection_count,
										TriangleData *triangle_data) {
	const int ni = grid.ni, nj = grid.nj, nk = grid.nk;
	const float dx = grid.dx;
	const Vec3f &origin = grid.origin;
	float phi_init = (float)(ni + nj + nk) * dx;
	uint32_t init_bits;
	memcpy(&init_bits, &phi_init, sizeof(init_bits));

	const uint64_t sentinel = ((uint64_t)init_bits << 32) | 0xffffffffu;
	const size_t ncells = (size_t)ni * nj * nk;
	std::atomic<uint64_t> *packed = (std::atomic<uint64_t> *)alloc_grid(
		ncells * sizeof(std::atomic<uint64_t>));
	assert(packed != NULL || ncells == 0);
#pragma omp parallel for
	for (ptrdiff_t c = 0; c < (ptrdiff_t)ncells; ++c) {
		new (&packed[c]) std::atomic<uint64_t>(sentinel);
	}
#pragma omp parallel
	{
#pragma omp for schedule(dynamic)
		for (int t = 0; t < ntri; ++t) {
			TriSetup setup = setup_triangle(tri[t], vertices, origin, dx);
			triangle_data[t] = setup.triangle;

			IndexRange band_i = band_range(setup.p_i, setup.q_i, setup.r_i, ni);
			IndexRange band_j = band_range(setup.p_j, setup.q_j, setup.r_j, nj);
			IndexRange band_k = band_range(setup.p_k, setup.q_k, setup.r_k, nk);
			for (int k = band_k.lo; k <= band_k.hi; ++k) {
				for (int j = band_j.lo; j <= band_j.hi; ++j) {
					for (int i = band_i.lo; i <= band_i.hi; ++i) {
						Vec3f gx((float)i * dx + origin.x,
								 (float)j * dx + origin.y,
								 (float)k * dx + origin.z);
						float d = point_triangle_distance(gx, setup.triangle);
						packed_min_distance(packed[i + ni * (j + nj * k)], d,
											(unsigned int)t);
					}
				}
			}

			IndexRange rows_j = row_range(setup.p_j, setup.q_j, setup.r_j, nj);
			IndexRange rows_k = row_range(setup.p_k, setup.q_k, setup.r_k, nk);
			for (int k = rows_k.lo; k <= rows_k.hi; ++k) {
				for (int j = rows_j.lo; j <= rows_j.hi; ++j) {
					double a, b, c;
					if (point_in_triangle_2d(j, k, setup.p_j, setup.p_k,
											 setup.q_j, setup.q_k, setup.r_j,
											 setup.r_k, a, b, c)) {
						double fi =
							a * setup.p_i + b * setup.q_i + c * setup.r_i;

						int i_interval = (int)ceil(fi);
						if (i_interval < 0) {
#pragma omp atomic
							++intersection_count(0, j, k);
						}
						else if (i_interval < ni) {
#pragma omp atomic
							++intersection_count(i_interval, j, k);
						}
					}
				}
			}
		}
	}
#pragma omp parallel for
	for (ptrdiff_t c = 0; c < (ptrdiff_t)ncells; ++c) {
		uint64_t w = packed[c].load(std::memory_order_relaxed);
		packed[c].~atomic();
		uint32_t bits = (uint32_t)(w >> 32);
		memcpy(&phi.storage[c], &bits, sizeof(bits));
		closest_tri.storage[c] = (int)(uint32_t)w;
	}
	free(packed);
}
