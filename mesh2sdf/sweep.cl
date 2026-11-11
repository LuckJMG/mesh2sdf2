// OpenCL sweep kernels for mesh2sdf/opencl.py. They port the sweep half of
// mesh2sdf/gpu.cu: the CPU runs init and signs (mesh2sdf.core.init and
// apply_signs), so this file needs only float32 and int32 math. opencl.py
// builds with -cl-fp32-correctly-rounded-divide-sqrt, and contraction into FMA
// is off, so results match the CPU path bit for bit. Keep the geometry in sync
// with csrc/geometry.h: tests/test_opencl.py compares them with array_equal.

#pragma OPENCL FP_CONTRACT OFF

typedef struct {
	float x, y, z;
} Vec3f;

// Same field order as the float32 rows from mesh2sdf.core.init.
typedef struct {
	Vec3f x1, x2, x3, x13, x23;
	float m13, m23, dot, invdet;
	Vec3f d12, d13, d23;
	float squared_length12, squared_length13, squared_length23;
} TriangleData;

inline Vec3f vec3f(float x, float y, float z) {
	Vec3f v = {x, y, z};
	return v;
}

inline Vec3f add(Vec3f a, Vec3f b) {
	return vec3f(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline Vec3f sub(Vec3f a, Vec3f b) {
	return vec3f(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline Vec3f scale(float s, Vec3f v) {
	return vec3f(s * v.x, s * v.y, s * v.z);
}

inline float vdot(Vec3f a, Vec3f b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

inline float dist(Vec3f a, Vec3f b) {
	float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
	return sqrt(dx * dx + dy * dy + dz * dz);
}

// csrc/geometry.h divides in double and rounds to float. Both operands are
// floats, so that equals one IEEE float division (see gpu.cu).
inline float point_segment_distance(Vec3f x0, Vec3f x1, Vec3f x2, Vec3f delta,
									float squared_length) {
	float s12 = vdot(sub(x2, x0), delta) / squared_length;
	if (s12 < 0) {
		s12 = 0;
	}
	else if (s12 > 1) {
		s12 = 1;
	}
	return dist(x0, add(scale(s12, x1), scale(1 - s12, x2)));
}

inline float point_triangle_distance(Vec3f x0,
									 __global const TriangleData *t) {
	Vec3f x03 = sub(x0, t->x3);
	float a = vdot(t->x13, x03), b = vdot(t->x23, x03);

	float w23 = t->invdet * (t->m23 * a - t->dot * b);
	float w31 = t->invdet * (t->m13 * b - t->dot * a);
	float w12 = 1 - w23 - w31;

	if (w23 >= 0 && w31 >= 0 && w12 >= 0) {
		return dist(x0, add(add(scale(w23, t->x1), scale(w31, t->x2)),
							scale(w12, t->x3)));
	}
	if (w23 > 0) {
		float d1 = point_segment_distance(x0, t->x1, t->x2, t->d12, t->squared_length12);
		float d2 = point_segment_distance(x0, t->x1, t->x3, t->d13, t->squared_length13);
		return d1 < d2 ? d1 : d2;
	}
	if (w31 > 0) {
		float d1 = point_segment_distance(x0, t->x1, t->x2, t->d12, t->squared_length12);
		float d2 = point_segment_distance(x0, t->x2, t->x3, t->d23, t->squared_length23);
		return d1 < d2 ? d1 : d2;
	}
	float d1 = point_segment_distance(x0, t->x1, t->x3, t->d13, t->squared_length13);
	float d2 = point_segment_distance(x0, t->x2, t->x3, t->d23, t->squared_length23);
	return d1 < d2 ? d1 : d2;
}

// Layouts: see the comment above skew() in gpu.cu.
inline long cell(int n, int i, int j, int k) {
	return (long)i + (long)n * ((long)j + (long)n * k);
}

inline int flip(int n, int d, int x) { return d > 0 ? x : n - 1 - x; }

inline long skew(int n, int slice, int b, int c) {
	return ((long)slice * n + c) * n + b;
}

inline long address(int n, int di, int dj, int dk, int i, int j, int k) {
	if (di == 0) {
		return cell(n, i, j, k);
	}
	int a = flip(n, di, i), b = flip(n, dj, j), c = flip(n, dk, k);
	return skew(n, (a + b + c) % n, b, c);
}

inline int pmod(int x, int n) { return (x % n + n) % n; }

// See relayout() in gpu.cu.
__kernel void relayout(__global const float *src_phi,
					   __global const int *src_tri,
					   __global const char *src_last, __global float *dst_phi,
					   __global int *dst_tri, __global char *dst_last, int n,
					   int sdi, int sdj, int sdk, int ddi, int ddj, int ddk) {
	int x = get_global_id(0), y = get_global_id(1), z = get_global_id(2);
	if (x >= n) {
		return;
	}
	long idx = cell(n, x, y, z);
	int i = x, j = y, k = z;
	if (ddi != 0) {
		int a = (z - x - y + 2 * n) % n;
		i = flip(n, ddi, a);
		j = flip(n, ddj, x);
		k = flip(n, ddk, y);
	}
	long src = address(n, sdi, sdj, sdk, i, j, k);
	dst_phi[idx] = src_phi[src];
	dst_tri[idx] = src_tri[src];
	dst_last[idx] = src_last[src];
}

// See relayout_flip_i() in gpu.cu. Work groups are 32 x 8.
__kernel void relayout_flip_i(__global const float *src_phi,
							  __global const int *src_tri,
							  __global const char *src_last,
							  __global float *dst_phi, __global int *dst_tri,
							  __global char *dst_last, int n, int sdi, int sdj,
							  int sdk, int ddi, int ddj, int ddk) {
	__local float tile_phi[32][33];
	__local int tile_tri[32][33];
	__local char tile_last[32][33];
	int tx = get_local_id(0), ty = get_local_id(1);
	int b0 = get_group_id(0) * 32, s0 = get_group_id(1) * 32;
	int c = get_group_id(2);
	int b = b0 + tx;
	int cs = flip(n, sdk, flip(n, ddk, c));

	// -s + 2b is smallest at s = s0 + 31, b = b0
	int a_first = flip(n, sdi, flip(n, ddi, pmod(s0 + 31 - b0 - c, n)));
	int first_slice = a_first + b0 + cs;
	if (b < n) {
		for (int r = ty; r < 3 * 32 - 2; r += 8) {
			int ss = pmod(first_slice + r, n);
			int a = flip(n, ddi, flip(n, sdi, pmod(ss - b - cs, n)));
			int row = pmod(a + b + c, n) - s0;
			if (row >= 0 && row < 32) {
				long src = skew(n, ss, b, cs);
				tile_phi[row][tx] = src_phi[src];
				tile_tri[row][tx] = src_tri[src];
				tile_last[row][tx] = src_last[src];
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int row = ty; row < 32; row += 8) {
		if (s0 + row < n && b < n) {
			long dst = skew(n, s0 + row, b, c);
			dst_phi[dst] = tile_phi[row][tx];
			dst_tri[dst] = tile_tri[row][tx];
			dst_last[dst] = tile_last[row][tx];
		}
	}
}

inline bool check_neighbour(__global const TriangleData *triangle_data,
							__global float *phi, __global int *closest_tri,
							Vec3f gx, long c0, long c1) {
	int t = closest_tri[c1];
	if (t >= 0 && t != closest_tri[c0]) {
		float d = point_triangle_distance(gx, triangle_data + t);
		if (d < phi[c0]) {
			phi[c0] = d;
			closest_tri[c0] = t;
			return true;
		}
	}
	return false;
}

// See sweep_plane() in gpu.cu. limits holds 7 entries per sweep. The plane
// comes in as the global offset of dimension 2, because a launch with a new
// offset costs much less than setting a kernel argument again.
__kernel void sweep_plane(__global const TriangleData *triangle_data,
						  __global float *phi, __global int *closest_tri,
						  __global char *last_change,
						  __global const int *limits, float origin, float dx,
						  int n, int di, int dj, int dk, int sweep) {
	int b = get_global_id(0) + 1;
	int c = get_global_id(1) + 1;
	int plane = get_global_id(2);
	int a = plane - b - c;
	if (b >= n || c >= n || a < 1 || a >= n) {
		return;
	}
	Vec3f gx = vec3f((float)flip(n, di, a) * dx + origin,
					 (float)flip(n, dj, b) * dx + origin,
					 (float)flip(n, dk, c) * dx + origin);
	bool interior = a <= n - 2 && b <= n - 2 && c <= n - 2;
	int slice0 = plane % n, slice1 = (plane - 1) % n;
	int slice2 = (plane - 2) % n, slice3 = (plane - 3) % n;
	long c0 = skew(n, slice0, b, c);
	long sources[7] = {
		skew(n, slice1, b, c),		   skew(n, slice1, b - 1, c),
		skew(n, slice2, b - 1, c),	   skew(n, slice1, b, c - 1),
		skew(n, slice2, b, c - 1),	   skew(n, slice2, b - 1, c - 1),
		skew(n, slice3, b - 1, c - 1)};
	__global const int *limit = limits + 7 * sweep;

	bool changed = false;
	for (int o = 0; o < 7; ++o) {
		if (!interior || last_change[sources[o]] > limit[o]) {
			changed |= check_neighbour(triangle_data, phi, closest_tri, gx, c0,
									   sources[o]);
		}
	}
	if (changed) {
		last_change[c0] = (char)sweep;
	}
}
