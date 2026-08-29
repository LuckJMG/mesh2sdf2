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

static void check_neighbour(const TriangleData *triangle_data, Array3f &phi,
							Array3i &closest_tri, const Vec3f &gx, int i0,
							int j0, int k0, int i1, int j1, int k1) {
	if (closest_tri(i1, j1, k1) >= 0) {
		const TriangleData &triangle = triangle_data[closest_tri(i1, j1, k1)];

		float d = point_triangle_distance(gx, triangle);
		if (d < phi(i0, j0, k0)) {
			phi(i0, j0, k0) = d;
			closest_tri(i0, j0, k0) = closest_tri(i1, j1, k1);
		}
	}
}

static void update_cell(const TriangleData *triangle_data, Array3f &phi,
						Array3i &closest_tri, const Vec3f &origin, float dx,
						int i, int j, int k, int di, int dj, int dk) {
	Vec3f gx((float)i * dx + origin.x, (float)j * dx + origin.y,
			 (float)k * dx + origin.z);

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

// Wavefront-parallel variant of sweep(). Cells are grouped by w = di*i + dj*j +
// dk*k. Every source read by a cell lies 1-3 levels behind it, and cells on one
// level never touch each other, so each level runs as one omp parallel for.
// Level order is a topological order of the same dependency DAG as the loop
// above, so results are bit-identical to the serial sweep at any thread count.
void sweep_parallel(const TriangleData *triangle_data, Array3f &phi,
					Array3i &closest_tri, const Vec3f &origin, float dx, int di,
					int dj, int dk) {
	const int i_lo = di > 0 ? 1 : 0;
	const int i_hi = di > 0 ? phi.ni - 1 : phi.ni - 2;

	const int j_lo = dj > 0 ? 1 : 0;
	const int j_hi = dj > 0 ? phi.nj - 1 : phi.nj - 2;

	const int k_lo = dk > 0 ? 1 : 0;
	const int k_hi = dk > 0 ? phi.nk - 1 : phi.nk - 2;

	int w_min = (di > 0 ? di * i_lo : di * i_hi) +
				(dj > 0 ? dj * j_lo : dj * j_hi) +
				(dk > 0 ? dk * k_lo : dk * k_hi);
	int w_max = (di > 0 ? di * i_hi : di * i_lo) +
				(dj > 0 ? dj * j_hi : dj * j_lo) +
				(dk > 0 ? dk * k_hi : dk * k_lo);
	for (int w = w_min; w <= w_max; ++w) {
#pragma omp parallel for schedule(static)
		for (int j = j_lo; j <= j_hi; ++j) {
			int m = w - dj * j;
			int t_lo = di > 0 ? i_lo : -i_hi;
			int t_hi = di > 0 ? i_hi : -i_lo;
			int a = m - t_hi, b = m - t_lo;

			int v1 = dk > 0 ? a : -b;
			int v2 = dk > 0 ? b : -a;

			int klo = k_lo > v1 ? k_lo : v1;
			int khi = k_hi < v2 ? k_hi : v2;
			for (int k = klo; k <= khi; ++k) {
				int i = di * (m - dk * k);
				update_cell(triangle_data, phi, closest_tri, origin, dx, i, j,
							k, di, dj, dk);
			}
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
// (diagonal 2*sqrt(3)), so untouched cells keep the sentinel low word and the
// unpack pass leaves their phi/closest_tri values alone.
void init_distances_and_counts_parallel(const Vec3ui *tri, int ntri,
										const Vec3f *x, const Vec3f &origin,
										float dx, int ni, int nj, int nk,
										const int exact_band, Array3f &phi,
										Array3i &closest_tri,
										Array3i &intersection_count,
										TriangleData *triangle_data) {
	float phi_init = (float)(ni + nj + nk) * dx;
	uint32_t init_bits;
	memcpy(&init_bits, &phi_init, sizeof(init_bits));

	const uint64_t sentinel = ((uint64_t)init_bits << 32) | 0xffffffffu;
	const size_t ncells = (size_t)ni * nj * nk;
	std::atomic<uint64_t> *packed =
		(std::atomic<uint64_t> *)malloc(ncells * sizeof(std::atomic<uint64_t>));
	assert(packed != NULL || ncells == 0);
	for (size_t c = 0; c < ncells; ++c) {
		new (&packed[c]) std::atomic<uint64_t>(sentinel);
	}
#pragma omp parallel
	{
#pragma omp for schedule(dynamic)
		for (int t = 0; t < ntri; ++t) {
			TriSetup setup = setup_triangle(tri[t], x, origin, dx);
			triangle_data[t] = setup.triangle;

			int i0 = clamp_int((int)min3(setup.p_i, setup.q_i, setup.r_i) -
								   exact_band,
							   0, ni - 1),
				i1 = clamp_int((int)max3(setup.p_i, setup.q_i, setup.r_i) +
								   exact_band + 1,
							   0, ni - 1);

			int j0 = clamp_int((int)min3(setup.p_j, setup.q_j, setup.r_j) -
								   exact_band,
							   0, nj - 1),

				j1 = clamp_int((int)max3(setup.p_j, setup.q_j, setup.r_j) +
								   exact_band + 1,
							   0, nj - 1);
			int k0 = clamp_int((int)min3(setup.p_k, setup.q_k, setup.r_k) -
								   exact_band,
							   0, nk - 1),
				k1 = clamp_int((int)max3(setup.p_k, setup.q_k, setup.r_k) +
								   exact_band + 1,
							   0, nk - 1);
			for (int k = k0; k <= k1; ++k) {
				for (int j = j0; j <= j1; ++j) {
					for (int i = i0; i <= i1; ++i) {
						Vec3f gx((float)i * dx + origin.x,
								 (float)j * dx + origin.y,
								 (float)k * dx + origin.z);
						float d = point_triangle_distance(gx, setup.triangle);
						packed_min_distance(packed[i + ni * (j + nj * k)], d,
											(unsigned int)t);
					}
				}
			}

			j0 = clamp_int((int)ceil(min3(setup.p_j, setup.q_j, setup.r_j)), 0,
						   nj - 1);
			j1 = clamp_int((int)floor(max3(setup.p_j, setup.q_j, setup.r_j)), 0,
						   nj - 1);

			k0 = clamp_int((int)ceil(min3(setup.p_k, setup.q_k, setup.r_k)), 0,
						   nk - 1);
			k1 = clamp_int((int)floor(max3(setup.p_k, setup.q_k, setup.r_k)), 0,
						   nk - 1);
			for (int k = k0; k <= k1; ++k) {
				for (int j = j0; j <= j1; ++j) {
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
	for (size_t c = 0; c < ncells; ++c) {
		uint64_t w = packed[c].load(std::memory_order_relaxed);
		packed[c].~atomic();
		if ((uint32_t)w != 0xffffffffu) {
			uint32_t bits = (uint32_t)(w >> 32);
			memcpy(&phi.storage[c], &bits, sizeof(bits));
			closest_tri.storage[c] = (int)(uint32_t)w;
		}
	}
	free(packed);
}
