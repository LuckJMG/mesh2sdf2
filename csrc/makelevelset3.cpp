#include "makelevelset3.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

struct SegmentData {
	Vec3f dx;
	double m2;
};

struct TriangleData {
	Vec3f x1, x2, x3;
	Vec3f x13, x23;
	float m13, m23, d, invdet;
	SegmentData edge12, edge13, edge23;
};

// find distance x0 is from segment x1-x2
static float point_segment_distance(const Vec3f &x0, const Vec3f &x1,
									const Vec3f &x2, const SegmentData &edge) {
	// find parameter value of closest point on segment
	float s12 = (float)(dot(x2 - x0, edge.dx) / edge.m2);
	if (s12 < 0) {
		s12 = 0;
	}
	else if (s12 > 1) {
		s12 = 1;
	}
	// and find the distance
	return dist(x0, s12 * x1 + (1 - s12) * x2);
}

// find distance x0 is from triangle x1-x2-x3
static float point_triangle_distance(const Vec3f &x0,
									 const TriangleData &triangle) {
	// first find barycentric coordinates of closest point on infinite plane
	Vec3f x03(x0 - triangle.x3);
	float a = dot(triangle.x13, x03), b = dot(triangle.x23, x03);
	// the barycentric coordinates themselves
	float w23 = triangle.invdet * (triangle.m23 * a - triangle.d * b);
	float w31 = triangle.invdet * (triangle.m13 * b - triangle.d * a);
	float w12 = 1 - w23 - w31;
	if (w23 >= 0 && w31 >= 0 && w12 >= 0) { // if we're inside the triangle
		return dist(x0,
					w23 * triangle.x1 + w31 * triangle.x2 + w12 * triangle.x3);
	}
	else {			   // we have to clamp to one of the edges
		if (w23 > 0) { // this rules out edge 2-3 for us
			return min(point_segment_distance(x0, triangle.x1, triangle.x2,
											  triangle.edge12),
					   point_segment_distance(x0, triangle.x1, triangle.x3,
											  triangle.edge13));
		}
		else if (w31 > 0) { // this rules out edge 1-3
			return min(point_segment_distance(x0, triangle.x1, triangle.x2,
											  triangle.edge12),
					   point_segment_distance(x0, triangle.x2, triangle.x3,
											  triangle.edge23));
		}
		else { // w12 must be >0, ruling out edge 1-2
			return min(point_segment_distance(x0, triangle.x1, triangle.x3,
											  triangle.edge13),
					   point_segment_distance(x0, triangle.x2, triangle.x3,
											  triangle.edge23));
		}
	}
}

static void check_neighbour(const std::vector<TriangleData> &triangle_data,
							Array3f &phi, Array3i &closest_tri, const Vec3f &gx,
							int i0, int j0, int k0, int i1, int j1, int k1) {
	if (closest_tri(i1, j1, k1) >= 0) {
		const TriangleData &triangle = triangle_data[closest_tri(i1, j1, k1)];
		float d = point_triangle_distance(gx, triangle);
		if (d < phi(i0, j0, k0)) {
			phi(i0, j0, k0) = d;
			closest_tri(i0, j0, k0) = closest_tri(i1, j1, k1);
		}
	}
}

static void update_cell(const std::vector<TriangleData> &triangle_data,
						Array3f &phi, Array3i &closest_tri, const Vec3f &origin,
						float dx, int i, int j, int k, int di, int dj, int dk) {
	Vec3f gx((float)i * dx + origin[0], (float)j * dx + origin[1],
			 (float)k * dx + origin[2]);
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

static void sweep(const std::vector<TriangleData> &triangle_data, Array3f &phi,
				  Array3i &closest_tri, const Vec3f &origin, float dx, int di,
				  int dj, int dk) {
	int i0, i1;
	if (di > 0) {
		i0 = 1;
		i1 = phi.ni;
	}
	else {
		i0 = phi.ni - 2;
		i1 = -1;
	}
	int j0, j1;
	if (dj > 0) {
		j0 = 1;
		j1 = phi.nj;
	}
	else {
		j0 = phi.nj - 2;
		j1 = -1;
	}
	int k0, k1;
	if (dk > 0) {
		k0 = 1;
		k1 = phi.nk;
	}
	else {
		k0 = phi.nk - 2;
		k1 = -1;
	}
	for (int k = k0; k != k1; k += dk) {
		for (int j = j0; j != j1; j += dj) {
			for (int i = i0; i != i1; i += di) {
				update_cell(triangle_data, phi, closest_tri, origin, dx, i, j,
							k, di, dj, dk);
			}
		}
	}
}

#ifdef _OPENMP
static constexpr size_t sweep_parallel_min_cells = 1 << 18;

// Wavefront-parallel variant of sweep(). Cells are grouped by w = di*i + dj*j +
// dk*k. Every source read by a cell lies 1-3 levels behind it, and cells on one
// level never touch each other, so each level runs as one omp parallel for.
// Level order is a topological order of the same dependency DAG as the loop
// above, so results are bit-identical to the serial sweep at any thread count.
static void sweep_parallel(const std::vector<TriangleData> &triangle_data,
						   Array3f &phi, Array3i &closest_tri,
						   const Vec3f &origin, float dx, int di, int dj,
						   int dk) {
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
			int klo = max(k_lo, dk > 0 ? a : -b);
			int khi = min(k_hi, dk > 0 ? b : -a);
			for (int k = klo; k <= khi; ++k) {
				int i = di * (m - dk * k);
				update_cell(triangle_data, phi, closest_tri, origin, dx, i, j,
							k, di, dj, dk);
			}
		}
	}
}
#endif

// calculate twice signed area of triangle (0,0)-(x1,y1)-(x2,y2)
// return an SOS-determined sign (-1, +1, or 0 only if it's a truly degenerate
// triangle)
static int orientation(double x1, double y1, double x2, double y2,
					   double &twice_signed_area) {
	twice_signed_area = y1 * x2 - x1 * y2;
	if (twice_signed_area > 0) {
		return 1;
	}
	if (twice_signed_area < 0) {
		return -1;
	}
	if (y2 > y1 || (y2 == y1 && x1 > x2)) {
		return 1;
	}
	if (y2 < y1 || (y2 == y1 && x1 < x2)) {
		return -1;
	}
	return 0; // only true when x1==x2 and y1==y2
}

// robust test of (x0,y0) in the triangle (x1,y1)-(x2,y2)-(x3,y3)
// if true is returned, the barycentric coordinates are set in a,b,c.
static bool point_in_triangle_2d(double x0, double y0, double x1, double y1,
								 double x2, double y2, double x3, double y3,
								 double &a, double &b, double &c) {
	x1 -= x0;
	x2 -= x0;
	x3 -= x0;
	y1 -= y0;
	y2 -= y0;
	y3 -= y0;
	int signa = orientation(x2, y2, x3, y3, a);
	if (signa == 0) {
		return false;
	}
	int signb = orientation(x3, y3, x1, y1, b);
	if (signb != signa) {
		return false;
	}
	int signc = orientation(x1, y1, x2, y2, c);
	if (signc != signa) {
		return false;
	}
	double sum = a + b + c;
	assert(sum != 0); // if the SOS signs match and are nonkero, there's no way
					  // all of a, b, and c are zero.
	a /= sum;
	b /= sum;
	c /= sum;
	return true;
}

// per-triangle precomputed data plus the triangle vertices in grid
// coordinates to high precision, shared by both init paths below
struct TriSetup {
	TriangleData triangle;
	double fip, fjp, fkp, fiq, fjq, fkq, fir, fjr, fkr;
};

static TriSetup setup_triangle(const Vec3ui &t, const std::vector<Vec3f> &x,
							   const Vec3f &origin, float dx) {
	unsigned int p, q, r;
	assign(t, p, q, r);
	TriSetup s;
	TriangleData &triangle = s.triangle;
	triangle.x1 = x[p];
	triangle.x2 = x[q];
	triangle.x3 = x[r];
	triangle.x13 = triangle.x1 - triangle.x3;
	triangle.x23 = triangle.x2 - triangle.x3;
	triangle.m13 = mag2(triangle.x13);
	triangle.m23 = mag2(triangle.x23);
	triangle.d = dot(triangle.x13, triangle.x23);
	triangle.invdet =
		1.f /
		max(triangle.m13 * triangle.m23 - triangle.d * triangle.d, 1e-30f);
	triangle.edge12 = {triangle.x2 - triangle.x1,
					   mag2(triangle.x2 - triangle.x1)};
	triangle.edge13 = {triangle.x3 - triangle.x1,
					   mag2(triangle.x3 - triangle.x1)};
	triangle.edge23 = {triangle.x3 - triangle.x2,
					   mag2(triangle.x3 - triangle.x2)};
	s.fip = ((double)x[p][0] - origin[0]) / dx;
	s.fjp = ((double)x[p][1] - origin[1]) / dx;
	s.fkp = ((double)x[p][2] - origin[2]) / dx;
	s.fiq = ((double)x[q][0] - origin[0]) / dx;
	s.fjq = ((double)x[q][1] - origin[1]) / dx;
	s.fkq = ((double)x[q][2] - origin[2]) / dx;
	s.fir = ((double)x[r][0] - origin[0]) / dx;
	s.fjr = ((double)x[r][1] - origin[1]) / dx;
	s.fkr = ((double)x[r][2] - origin[2]) / dx;
	return s;
}

#ifdef _OPENMP
static constexpr size_t init_parallel_min_tris = 1 << 12;

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
static void init_distances_and_counts_parallel(
	const std::vector<Vec3ui> &tri, const std::vector<Vec3f> &x,
	const Vec3f &origin, float dx, int ni, int nj, int nk, const int exact_band,
	Array3f &phi, Array3i &closest_tri, Array3i &intersection_count,
	std::vector<TriangleData> &triangle_data) {
	float phi_init = (float)(ni + nj + nk) * dx;
	uint32_t init_bits;
	memcpy(&init_bits, &phi_init, sizeof(init_bits));
	const uint64_t sentinel = ((uint64_t)init_bits << 32) | 0xffffffffu;
	const size_t ncells = (size_t)ni * nj * nk;
	std::unique_ptr<std::atomic<uint64_t>[]> packed(
		new std::atomic<uint64_t>[ncells]);
	for (size_t c = 0; c < ncells; ++c) {
		packed[c].store(sentinel, std::memory_order_relaxed);
	}
	triangle_data.resize(tri.size());
#pragma omp parallel
	{
#pragma omp for schedule(dynamic)
		for (int t = 0; t < (int)tri.size(); ++t) {
			TriSetup setup = setup_triangle(tri[t], x, origin, dx);
			triangle_data[t] = setup.triangle;
			// do distances nearby
			int i0 = clamp(int(min(setup.fip, setup.fiq, setup.fir)) -
							   exact_band,
						   0, ni - 1),
				i1 = clamp(int(max(setup.fip, setup.fiq, setup.fir)) +
							   exact_band + 1,
						   0, ni - 1);
			int j0 = clamp(int(min(setup.fjp, setup.fjq, setup.fjr)) -
							   exact_band,
						   0, nj - 1),
				j1 = clamp(int(max(setup.fjp, setup.fjq, setup.fjr)) +
							   exact_band + 1,
						   0, nj - 1);
			int k0 = clamp(int(min(setup.fkp, setup.fkq, setup.fkr)) -
							   exact_band,
						   0, nk - 1),
				k1 = clamp(int(max(setup.fkp, setup.fkq, setup.fkr)) +
							   exact_band + 1,
						   0, nk - 1);
			for (int k = k0; k <= k1; ++k) {
				for (int j = j0; j <= j1; ++j) {
					for (int i = i0; i <= i1; ++i) {
						Vec3f gx((float)i * dx + origin[0],
								 (float)j * dx + origin[1],
								 (float)k * dx + origin[2]);
						float d = point_triangle_distance(gx, setup.triangle);
						packed_min_distance(packed[i + ni * (j + nj * k)], d,
											(unsigned int)t);
					}
				}
			}
			// and do intersection counts
			j0 = clamp((int)std::ceil(min(setup.fjp, setup.fjq, setup.fjr)), 0,
					   nj - 1);
			j1 = clamp((int)std::floor(max(setup.fjp, setup.fjq, setup.fjr)), 0,
					   nj - 1);
			k0 = clamp((int)std::ceil(min(setup.fkp, setup.fkq, setup.fkr)), 0,
					   nk - 1);
			k1 = clamp((int)std::floor(max(setup.fkp, setup.fkq, setup.fkr)), 0,
					   nk - 1);
			for (int k = k0; k <= k1; ++k) {
				for (int j = j0; j <= j1; ++j) {
					double a, b, c;
					if (point_in_triangle_2d(j, k, setup.fjp, setup.fkp,
											 setup.fjq, setup.fkq, setup.fjr,
											 setup.fkr, a, b, c)) {
						double fi = a * setup.fip + b * setup.fiq +
									c * setup.fir; // intersection i coordinate
						int i_interval =
							int(std::ceil(fi)); // intersection is in
												// (i_interval-1,i_interval]
						if (i_interval < 0) {
#pragma omp atomic
							++intersection_count(
								0, j,
								k); // we enlarge the first interval to
									// include everything to the -x direction
						}
						else if (i_interval < ni) {
#pragma omp atomic
							++intersection_count(i_interval, j, k);
						}
						// we ignore intersections that are beyond the +x side
						// of the grid
					}
				}
			}
		}
	}
	for (size_t c = 0; c < ncells; ++c) {
		uint64_t w = packed[c].load(std::memory_order_relaxed);
		if ((uint32_t)w != 0xffffffffu) {
			uint32_t bits = (uint32_t)(w >> 32);
			memcpy(&phi.a[c], &bits, sizeof(bits));
			closest_tri.a[c] = (int)(uint32_t)w;
		}
	}
}
#endif

// initialize distances near the mesh within exact_band cells of each triangle,
// and record triangle intersections along each grid row for later sign
// determination
static void init_distances_and_counts(
	const std::vector<Vec3ui> &tri, const std::vector<Vec3f> &x,
	const Vec3f &origin, float dx, int ni, int nj, int nk, const int exact_band,
	Array3f &phi, Array3i &closest_tri, Array3i &intersection_count,
	std::vector<TriangleData> &triangle_data) {
#ifdef _OPENMP
	if (omp_get_max_threads() > 1 && tri.size() >= init_parallel_min_tris) {
		init_distances_and_counts_parallel(tri, x, origin, dx, ni, nj, nk,
										   exact_band, phi, closest_tri,
										   intersection_count, triangle_data);
		return;
	}
#endif
	triangle_data.resize(tri.size());
	for (unsigned int t = 0; t < tri.size(); ++t) {
		TriSetup setup = setup_triangle(tri[t], x, origin, dx);
		triangle_data[t] = setup.triangle;
		// do distances nearby
		int i0 = clamp(int(min(setup.fip, setup.fiq, setup.fir)) - exact_band,
					   0, ni - 1),
			i1 = clamp(int(max(setup.fip, setup.fiq, setup.fir)) + exact_band +
						   1,
					   0, ni - 1);
		int j0 = clamp(int(min(setup.fjp, setup.fjq, setup.fjr)) - exact_band,
					   0, nj - 1),
			j1 = clamp(int(max(setup.fjp, setup.fjq, setup.fjr)) + exact_band +
						   1,
					   0, nj - 1);
		int k0 = clamp(int(min(setup.fkp, setup.fkq, setup.fkr)) - exact_band,
					   0, nk - 1),
			k1 = clamp(int(max(setup.fkp, setup.fkq, setup.fkr)) + exact_band +
						   1,
					   0, nk - 1);
		for (int k = k0; k <= k1; ++k) {
			for (int j = j0; j <= j1; ++j) {
				for (int i = i0; i <= i1; ++i) {
					Vec3f gx((float)i * dx + origin[0],
							 (float)j * dx + origin[1],
							 (float)k * dx + origin[2]);
					float d = point_triangle_distance(gx, setup.triangle);
					if (d < phi(i, j, k)) {
						phi(i, j, k) = d;
						closest_tri(i, j, k) = (int)t;
					}
				}
			}
		}
		// and do intersection counts
		j0 = clamp((int)std::ceil(min(setup.fjp, setup.fjq, setup.fjr)), 0,
				   nj - 1);
		j1 = clamp((int)std::floor(max(setup.fjp, setup.fjq, setup.fjr)), 0,
				   nj - 1);
		k0 = clamp((int)std::ceil(min(setup.fkp, setup.fkq, setup.fkr)), 0,
				   nk - 1);
		k1 = clamp((int)std::floor(max(setup.fkp, setup.fkq, setup.fkr)), 0,
				   nk - 1);
		for (int k = k0; k <= k1; ++k) {
			for (int j = j0; j <= j1; ++j) {
				double a, b, c;
				if (point_in_triangle_2d(j, k, setup.fjp, setup.fkp, setup.fjq,
										 setup.fkq, setup.fjr, setup.fkr, a, b,
										 c)) {
					double fi = a * setup.fip + b * setup.fiq +
								c * setup.fir; // intersection i coordinate
					int i_interval = int(std::ceil(
						fi)); // intersection is in (i_interval-1,i_interval]
					if (i_interval < 0) {
						++intersection_count(
							0, j, k); // we enlarge the first interval to
									  // include everything to the -x direction
					}
					else if (i_interval < ni) {
						++intersection_count(i_interval, j, k);
					}
					// we ignore intersections that are beyond the +x side of
					// the grid
				}
			}
		}
	}
}

// fill in the rest of the distances with fast sweeping
static void
run_fast_sweeping_passes(const std::vector<TriangleData> &triangle_data,
						 Array3f &phi, Array3i &closest_tri,
						 const Vec3f &origin, float dx) {
	void (*sweep_once)(const std::vector<TriangleData> &, Array3f &, Array3i &,
					   const Vec3f &, float, int, int, int) = sweep;
#ifdef _OPENMP
	if (omp_get_max_threads() > 1 &&
		(size_t)phi.ni * phi.nj * phi.nk >= sweep_parallel_min_cells) {
		sweep_once = sweep_parallel;
	}
#endif
	for (unsigned int pass = 0; pass < 2; ++pass) {
		sweep_once(triangle_data, phi, closest_tri, origin, dx, +1, +1, +1);
		sweep_once(triangle_data, phi, closest_tri, origin, dx, -1, -1, -1);
		sweep_once(triangle_data, phi, closest_tri, origin, dx, +1, +1, -1);
		sweep_once(triangle_data, phi, closest_tri, origin, dx, -1, -1, +1);
		sweep_once(triangle_data, phi, closest_tri, origin, dx, +1, -1, +1);
		sweep_once(triangle_data, phi, closest_tri, origin, dx, -1, +1, -1);
		sweep_once(triangle_data, phi, closest_tri, origin, dx, +1, -1, -1);
		sweep_once(triangle_data, phi, closest_tri, origin, dx, -1, +1, +1);
	}
}

// figure out signs (inside/outside) from intersection counts
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

void make_level_set3(const std::vector<Vec3ui> &tri,
					 const std::vector<Vec3f> &x, const Vec3f &origin, float dx,
					 int ni, int nj, int nk, Array3f &phi,
					 const int exact_band) {
	phi.resize(ni, nj, nk);
	phi.assign((float)(ni + nj + nk) * dx); // upper bound on distance
	Array3i closest_tri(ni, nj, nk, -1);
	Array3i intersection_count(ni, nj, nk,
							   0); // intersection_count(i,j,k) is # of tri
								   // intersections in (i-1,i]x{j}x{k}
	std::vector<TriangleData> triangle_data;
	init_distances_and_counts(tri, x, origin, dx, ni, nj, nk, exact_band, phi,
							  closest_tri, intersection_count, triangle_data);
	run_fast_sweeping_passes(triangle_data, phi, closest_tri, origin, dx);
	apply_signs_from_intersection_counts(intersection_count, phi, ni, nj, nk);
}
