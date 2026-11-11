// CUDA kernels for mesh2sdf/gpu.py, compiled at runtime by NVRTC through CuPy.
// The geometry copies csrc/vec.h and csrc/geometry.h operation for operation,
// and gpu.py compiles with --fmad=false, so results match the CPU path bit for
// bit. Keep both copies in sync: tests/test_gpu.py compares them with
// array_equal.

struct Vec3f {
	float x, y, z;
};

__device__ inline Vec3f vec3f(float x, float y, float z) {
	Vec3f v = {x, y, z};
	return v;
}

__device__ inline Vec3f operator+(const Vec3f &a, const Vec3f &b) {
	return vec3f(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline Vec3f operator-(const Vec3f &a, const Vec3f &b) {
	return vec3f(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline Vec3f operator*(float s, const Vec3f &v) {
	return vec3f(s * v.x, s * v.y, s * v.z);
}

__device__ inline float dot(const Vec3f &a, const Vec3f &b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float mag2(const Vec3f &a) { return dot(a, a); }

__device__ inline float dist(const Vec3f &a, const Vec3f &b) {
	float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
	return sqrtf(dx * dx + dy * dy + dz * dz);
}

struct SegmentData {
	Vec3f delta;
	double squared_length;
};

struct TriangleData {
	Vec3f x1, x2, x3;
	Vec3f x13, x23;
	float m13, m23, dot, invdet;
	SegmentData edge12, edge13, edge23;
};

// gpu.py allocates triangle data with this size
static_assert(sizeof(TriangleData) == 152, "update TRIANGLE_DATA_BYTES");

static const int exact_band = 1;

__device__ inline int clamp_int(int v, int lo, int hi) {
	return v < lo ? lo : (v > hi ? hi : v);
}

__device__ inline double min3(double a, double b, double c) {
	double m = a;
	if (b < m) {
		m = b;
	}
	if (c < m) {
		m = c;
	}
	return m;
}

__device__ inline double max3(double a, double b, double c) {
	double m = a;
	if (b > m) {
		m = b;
	}
	if (c > m) {
		m = c;
	}
	return m;
}

struct IndexRange {
	int lo, hi;
};

__device__ inline IndexRange band_range(double p, double q, double r, int n) {
	IndexRange range = {
		clamp_int((int)min3(p, q, r) - exact_band, 0, n - 1),
		clamp_int((int)max3(p, q, r) + exact_band + 1, 0, n - 1)};
	return range;
}

__device__ inline IndexRange row_range(double p, double q, double r, int n) {
	IndexRange range = {clamp_int((int)ceil(min3(p, q, r)), 0, n - 1),
						clamp_int((int)floor(max3(p, q, r)), 0, n - 1)};
	return range;
}

__device__ inline float point_segment_distance(const Vec3f &x0,
											   const Vec3f &x1,
											   const Vec3f &x2,
											   const SegmentData &edge) {
	// csrc/geometry.h divides in double and rounds to float. Both operands are
	// floats and double has at least 2 * 24 + 2 bits, so that double rounding
	// equals one IEEE float division (Figueroa, "When is double rounding
	// innocuous?", 1995). Float division avoids slow FP64 units on most GPUs.
	float s12 = dot(x2 - x0, edge.delta) / (float)edge.squared_length;
	if (s12 < 0) {
		s12 = 0;
	}
	else if (s12 > 1) {
		s12 = 1;
	}
	return dist(x0, s12 * x1 + (1 - s12) * x2);
}

__device__ inline float point_triangle_distance(const Vec3f &x0,
												const TriangleData &triangle) {
	Vec3f x03(x0 - triangle.x3);
	float a = dot(triangle.x13, x03), b = dot(triangle.x23, x03);

	float w23 = triangle.invdet * (triangle.m23 * a - triangle.dot * b);
	float w31 = triangle.invdet * (triangle.m13 * b - triangle.dot * a);
	float w12 = 1 - w23 - w31;

	if (w23 >= 0 && w31 >= 0 && w12 >= 0) {
		return dist(x0,
					w23 * triangle.x1 + w31 * triangle.x2 + w12 * triangle.x3);
	}
	if (w23 > 0) {
		float d1 = point_segment_distance(x0, triangle.x1, triangle.x2,
										  triangle.edge12);
		float d2 = point_segment_distance(x0, triangle.x1, triangle.x3,
										  triangle.edge13);
		return d1 < d2 ? d1 : d2;
	}
	if (w31 > 0) {
		float d1 = point_segment_distance(x0, triangle.x1, triangle.x2,
										  triangle.edge12);
		float d2 = point_segment_distance(x0, triangle.x2, triangle.x3,
										  triangle.edge23);
		return d1 < d2 ? d1 : d2;
	}

	float d1 =
		point_segment_distance(x0, triangle.x1, triangle.x3, triangle.edge13);
	float d2 =
		point_segment_distance(x0, triangle.x2, triangle.x3, triangle.edge23);
	return d1 < d2 ? d1 : d2;
}

__device__ inline int orientation(double x1, double y1, double x2, double y2,
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
	return 0;
}

__device__ inline bool point_in_triangle_2d(double x0, double y0, double x1,
											double y1, double x2, double y2,
											double x3, double y3, double &a,
											double &b, double &c) {
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
	a /= sum;
	b /= sum;
	c /= sum;
	return true;
}

struct TriSetup {
	TriangleData triangle;
	double p_i, p_j, p_k;
	double q_i, q_j, q_k;
	double r_i, r_j, r_k;
};

__device__ inline Vec3f vertex(const float *vertices, unsigned int v) {
	return vec3f(vertices[3 * v], vertices[3 * v + 1], vertices[3 * v + 2]);
}

__device__ inline TriSetup setup_triangle(const unsigned int *face,
										  const float *vertices, float origin,
										  float dx) {
	TriSetup s;
	TriangleData &triangle = s.triangle;
	Vec3f vp = vertex(vertices, face[0]);
	Vec3f vq = vertex(vertices, face[1]);
	Vec3f vr = vertex(vertices, face[2]);
	triangle.x1 = vp;
	triangle.x2 = vq;
	triangle.x3 = vr;
	triangle.x13 = triangle.x1 - triangle.x3;
	triangle.x23 = triangle.x2 - triangle.x3;
	triangle.m13 = mag2(triangle.x13);
	triangle.m23 = mag2(triangle.x23);
	triangle.dot = dot(triangle.x13, triangle.x23);

	double det = (double)triangle.m13 * triangle.m23 -
				 (double)triangle.dot * triangle.dot;
	double det_clamped = det > 1e-30 ? det : 1e-30;

	triangle.invdet = (float)(1.0 / det_clamped);
	triangle.edge12.delta = triangle.x2 - triangle.x1;
	triangle.edge12.squared_length = (double)mag2(triangle.x2 - triangle.x1);
	triangle.edge13.delta = triangle.x3 - triangle.x1;
	triangle.edge13.squared_length = (double)mag2(triangle.x3 - triangle.x1);
	triangle.edge23.delta = triangle.x3 - triangle.x2;
	triangle.edge23.squared_length = (double)mag2(triangle.x3 - triangle.x2);

	s.p_i = ((double)vp.x - origin) / dx;
	s.p_j = ((double)vp.y - origin) / dx;
	s.p_k = ((double)vp.z - origin) / dx;
	s.q_i = ((double)vq.x - origin) / dx;
	s.q_j = ((double)vq.y - origin) / dx;
	s.q_k = ((double)vq.z - origin) / dx;
	s.r_i = ((double)vr.x - origin) / dx;
	s.r_j = ((double)vr.y - origin) / dx;
	s.r_k = ((double)vr.z - origin) / dx;
	return s;
}

__device__ inline long long cell(int n, int i, int j, int k) {
	return (long long)i + (long long)n * ((long long)j + (long long)n * k);
}

// The sweeps run on a skewed layout. In sweep coordinates a, b, c (each counts
// from the start corner of a sweep), cell (a, b, c) lives in slice
// (a + b + c) mod n, row c, column b. This maps the n^3 cells one to one, so
// the layout needs no extra memory. Plane a + b + c of a sweep fills part of
// one slice, and threads with consecutive b touch consecutive addresses.
__device__ inline int flip(int n, int d, int x) { return d > 0 ? x : n - 1 - x; }

__device__ inline long long skew(int n, int slice, int b, int c) {
	return ((long long)slice * n + c) * n + b;
}

// Address of cell (i, j, k) in the layout of sweep (di, dj, dk). di == 0
// names the standard layout of csrc/array3.h.
__device__ inline long long address(int n, int di, int dj, int dk, int i,
									int j, int k) {
	if (di == 0) {
		return cell(n, i, j, k);
	}
	int a = flip(n, di, i), b = flip(n, dj, j), c = flip(n, dk, k);
	return skew(n, (a + b + c) % n, b, c);
}

// One block per triangle. Same packed minimum as
// init_distances_and_counts_parallel in csrc/parallel.cpp: the high word holds
// the non-negative distance bits and the low word the triangle index, so
// atomicMin keeps the smallest distance and, on ties, the smallest index.
// packed uses the layout of the first sweep (+1, +1, +1), so the sweeps start
// without a relayout. intersection_count uses the standard layout.
extern "C" __global__ void
init_distances_and_counts(const float *vertices, const unsigned int *faces,
						  float origin, float dx, int n,
						  unsigned long long *packed, int *intersection_count,
						  TriangleData *triangle_data) {
	__shared__ TriSetup setup;
	const int t = blockIdx.x;
	if (threadIdx.x == 0) {
		setup = setup_triangle(faces + 3 * t, vertices, origin, dx);
		triangle_data[t] = setup.triangle;
	}
	__syncthreads();

	IndexRange band_i = band_range(setup.p_i, setup.q_i, setup.r_i, n);
	IndexRange band_j = band_range(setup.p_j, setup.q_j, setup.r_j, n);
	IndexRange band_k = band_range(setup.p_k, setup.q_k, setup.r_k, n);
	int wi = band_i.hi - band_i.lo + 1, wj = band_j.hi - band_j.lo + 1;
	long long band_cells = (long long)wi * wj * (band_k.hi - band_k.lo + 1);
	for (long long c = threadIdx.x; c < band_cells; c += blockDim.x) {
		int i = band_i.lo + (int)(c % wi);
		int j = band_j.lo + (int)(c / wi % wj);
		int k = band_k.lo + (int)(c / ((long long)wi * wj));
		Vec3f gx = vec3f((float)i * dx + origin, (float)j * dx + origin,
						 (float)k * dx + origin);
		float d = point_triangle_distance(gx, setup.triangle);
		unsigned long long w =
			((unsigned long long)__float_as_uint(d) << 32) | (unsigned int)t;
		atomicMin(&packed[address(n, 1, 1, 1, i, j, k)], w);
	}

	IndexRange rows_j = row_range(setup.p_j, setup.q_j, setup.r_j, n);
	IndexRange rows_k = row_range(setup.p_k, setup.q_k, setup.r_k, n);
	int rj = rows_j.hi - rows_j.lo + 1, rk = rows_k.hi - rows_k.lo + 1;
	if (rj <= 0 || rk <= 0) {
		return;
	}
	for (int r = threadIdx.x; r < rj * rk; r += blockDim.x) {
		int j = rows_j.lo + r % rj, k = rows_k.lo + r / rj;
		double a, b, c;
		if (point_in_triangle_2d(j, k, setup.p_j, setup.p_k, setup.q_j,
								 setup.q_k, setup.r_j, setup.r_k, a, b, c)) {
			double fi = a * setup.p_i + b * setup.q_i + c * setup.r_i;
			int i_interval = (int)ceil(fi);
			if (i_interval < 0) {
				atomicAdd(&intersection_count[cell(n, 0, j, k)], 1);
			}
			else if (i_interval < n) {
				atomicAdd(&intersection_count[cell(n, i_interval, j, k)], 1);
			}
		}
	}
}

__device__ inline bool check_neighbour(const TriangleData *triangle_data,
									   float *phi, int *closest_tri,
									   const Vec3f &gx, long long c0,
									   long long c1) {
	int t = closest_tri[c1];
	if (t >= 0 && t != closest_tri[c0]) {
		float d = point_triangle_distance(gx, triangle_data[t]);
		if (d < phi[c0]) {
			phi[c0] = d;
			closest_tri[c0] = t;
			return true;
		}
	}
	return false;
}

// Gathers phi, closest_tri and last_change from layout s into layout d. One
// thread per destination address (x, y, z), so the writes coalesce.
extern "C" __global__ void relayout(const float *src_phi, const int *src_tri,
									const signed char *src_last,
									float *dst_phi, int *dst_tri,
									signed char *dst_last, int n, int sdi,
									int sdj, int sdk, int ddi, int ddj,
									int ddk) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y, z = blockIdx.z;
	if (x >= n) {
		return;
	}
	long long idx = cell(n, x, y, z);
	int i = x, j = y, k = z;
	if (ddi != 0) {
		int a = (z - x - y + 2 * n) % n;
		i = flip(n, ddi, a);
		j = flip(n, ddj, x);
		k = flip(n, ddk, y);
	}
	long long src = address(n, sdi, sdj, sdk, i, j, k);
	dst_phi[idx] = src_phi[src];
	dst_tri[idx] = src_tri[src];
	dst_last[idx] = src_last[src];
}

__device__ inline int pmod(int x, int n) { return (x % n + n) % n; }

// Relayout between two sweeps that differ in di but not in dj. relayout()
// reads two source slices apart per thread there, so its reads do not
// coalesce. Here each 32 x 8 block fills a 32 x 32 tile (slices s0..s0+31,
// columns b0..b0+31) of one destination row c through shared memory. The
// source slice of (s, b) is -s + 2b plus a constant (mod n), so the tile reads
// 3 * 32 - 2 consecutive source slices, each along 32 consecutive columns.
extern "C" __global__ void
relayout_flip_i(const float *src_phi, const int *src_tri,
				const signed char *src_last, float *dst_phi, int *dst_tri,
				signed char *dst_last, int n, int sdi, int sdj, int sdk,
				int ddi, int ddj, int ddk) {
	__shared__ float tile_phi[32][33];
	__shared__ int tile_tri[32][33];
	__shared__ signed char tile_last[32][33];
	int b0 = blockIdx.x * 32, s0 = blockIdx.y * 32, c = blockIdx.z;
	int b = b0 + threadIdx.x;
	int cs = flip(n, sdk, flip(n, ddk, c));

	// source slice of (s0 + 31, b0), the first one the tile needs
	int a_first = flip(n, sdi, flip(n, ddi, pmod(s0 + 31 - b0 - c, n)));
	int first = a_first + b0 + cs;
	if (b < n) {
		for (int r = threadIdx.y; r < 3 * 32 - 2; r += blockDim.y) {
			int ss = pmod(first + r, n);
			int a = flip(n, ddi, flip(n, sdi, pmod(ss - b - cs, n)));
			int row = pmod(a + b + c, n) - s0;
			if (row >= 0 && row < 32) {
				long long src = skew(n, ss, b, cs);
				tile_phi[row][threadIdx.x] = src_phi[src];
				tile_tri[row][threadIdx.x] = src_tri[src];
				tile_last[row][threadIdx.x] = src_last[src];
			}
		}
	}
	__syncthreads();
	for (int row = threadIdx.y; row < 32; row += blockDim.y) {
		if (s0 + row < n && b < n) {
			long long dst = skew(n, s0 + row, b, c);
			dst_phi[dst] = tile_phi[row][threadIdx.x];
			dst_tri[dst] = tile_tri[row][threadIdx.x];
			dst_last[dst] = tile_last[row][threadIdx.x];
		}
	}
}

// One plane of a sweep, on the skewed layout of that sweep. A cell reads its 7
// neighbours at a-1, b-1, c-1, which lie on the 1-3 planes before plane
// a + b + c. slice0..slice3 are the slices of planes plane..plane-3.
// Launching one plane after another walks a topological order of the same
// dependency graph as sweep() in csrc/makelevelset3.cpp, so the results are
// bit-identical.
//
// The per-check form of the row skip in csrc/makelevelset3.cpp. last_change
// holds, per cell and in the same layout as phi, the last sweep that changed
// the cell, or -1. limit[o] is the last earlier sweep that ran check o on
// every interior cell, or -2. If the source cell did not change after that
// sweep, the check repeats a triangle that already lost, and phi only
// decreases, so the check cannot change the cell. Cells on the boundary do not
// run in every sweep, so they never skip.
extern "C" __global__ void
sweep_plane(const TriangleData *triangle_data, float *phi, int *closest_tri,
			signed char *last_change, const int *limit, float origin,
			float dx, int n, int di, int dj, int dk, int sweep, int plane,
			int slice0, int slice1, int slice2, int slice3) {
	int b = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int c = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int a = plane - b - c;
	if (b >= n || c >= n || a < 1 || a >= n) {
		return;
	}
	Vec3f gx = vec3f((float)flip(n, di, a) * dx + origin,
					 (float)flip(n, dj, b) * dx + origin,
					 (float)flip(n, dk, c) * dx + origin);
	bool interior = a <= n - 2 && b <= n - 2 && c <= n - 2;
	long long c0 = skew(n, slice0, b, c);
	long long sources[7] = {
		skew(n, slice1, b, c),		   skew(n, slice1, b - 1, c),
		skew(n, slice2, b - 1, c),	   skew(n, slice1, b, c - 1),
		skew(n, slice2, b, c - 1),	   skew(n, slice2, b - 1, c - 1),
		skew(n, slice3, b - 1, c - 1)};

	bool changed = false;
	for (int o = 0; o < 7; ++o) {
		if (!interior || last_change[sources[o]] > limit[o]) {
			changed |= check_neighbour(triangle_data, phi, closest_tri, gx, c0,
									   sources[o]);
		}
	}
	if (changed) {
		last_change[c0] = (signed char)sweep;
	}
}
