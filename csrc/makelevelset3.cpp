#include "makelevelset3.h"

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
				Vec3f gx((float)i * dx + origin[0], (float)j * dx + origin[1],
						 (float)k * dx + origin[2]);
				check_neighbour(triangle_data, phi, closest_tri, gx, i, j, k,
								i - di, j, k);
				check_neighbour(triangle_data, phi, closest_tri, gx, i, j, k, i,
								j - dj, k);
				check_neighbour(triangle_data, phi, closest_tri, gx, i, j, k,
								i - di, j - dj, k);
				check_neighbour(triangle_data, phi, closest_tri, gx, i, j, k, i,
								j, k - dk);
				check_neighbour(triangle_data, phi, closest_tri, gx, i, j, k,
								i - di, j, k - dk);
				check_neighbour(triangle_data, phi, closest_tri, gx, i, j, k, i,
								j - dj, k - dk);
				check_neighbour(triangle_data, phi, closest_tri, gx, i, j, k,
								i - di, j - dj, k - dk);
			}
		}
	}
}

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

// initialize distances near the mesh within exact_band cells of each triangle,
// and record triangle intersections along each grid row for later sign
// determination
static void init_distances_and_counts(
	const std::vector<Vec3ui> &tri, const std::vector<Vec3f> &x,
	const Vec3f &origin, float dx, int ni, int nj, int nk, const int exact_band,
	Array3f &phi, Array3i &closest_tri, Array3i &intersection_count,
	std::vector<TriangleData> &triangle_data) {
	triangle_data.resize(tri.size());
	for (unsigned int t = 0; t < tri.size(); ++t) {
		unsigned int p, q, r;
		assign(tri[t], p, q, r);
		TriangleData &triangle = triangle_data[t];
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
		// coordinates in grid to high precision
		double fip = ((double)x[p][0] - origin[0]) / dx,
			   fjp = ((double)x[p][1] - origin[1]) / dx,
			   fkp = ((double)x[p][2] - origin[2]) / dx;
		double fiq = ((double)x[q][0] - origin[0]) / dx,
			   fjq = ((double)x[q][1] - origin[1]) / dx,
			   fkq = ((double)x[q][2] - origin[2]) / dx;
		double fir = ((double)x[r][0] - origin[0]) / dx,
			   fjr = ((double)x[r][1] - origin[1]) / dx,
			   fkr = ((double)x[r][2] - origin[2]) / dx;
		// do distances nearby
		int i0 = clamp(int(min(fip, fiq, fir)) - exact_band, 0, ni - 1),
			i1 = clamp(int(max(fip, fiq, fir)) + exact_band + 1, 0, ni - 1);
		int j0 = clamp(int(min(fjp, fjq, fjr)) - exact_band, 0, nj - 1),
			j1 = clamp(int(max(fjp, fjq, fjr)) + exact_band + 1, 0, nj - 1);
		int k0 = clamp(int(min(fkp, fkq, fkr)) - exact_band, 0, nk - 1),
			k1 = clamp(int(max(fkp, fkq, fkr)) + exact_band + 1, 0, nk - 1);
		for (int k = k0; k <= k1; ++k) {
			for (int j = j0; j <= j1; ++j) {
				for (int i = i0; i <= i1; ++i) {
					Vec3f gx((float)i * dx + origin[0],
							 (float)j * dx + origin[1],
							 (float)k * dx + origin[2]);
					float d = point_triangle_distance(gx, triangle);
					if (d < phi(i, j, k)) {
						phi(i, j, k) = d;
						closest_tri(i, j, k) = (int)t;
					}
				}
			}
		}
		// and do intersection counts
		j0 = clamp((int)std::ceil(min(fjp, fjq, fjr)), 0, nj - 1);
		j1 = clamp((int)std::floor(max(fjp, fjq, fjr)), 0, nj - 1);
		k0 = clamp((int)std::ceil(min(fkp, fkq, fkr)), 0, nk - 1);
		k1 = clamp((int)std::floor(max(fkp, fkq, fkr)), 0, nk - 1);
		for (int k = k0; k <= k1; ++k) {
			for (int j = j0; j <= j1; ++j) {
				double a, b, c;
				if (point_in_triangle_2d(j, k, fjp, fkp, fjq, fkq, fjr, fkr, a,
										 b, c)) {
					double fi = a * fip + b * fiq +
								c * fir; // intersection i coordinate
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
	for (unsigned int pass = 0; pass < 2; ++pass) {
		sweep(triangle_data, phi, closest_tri, origin, dx, +1, +1, +1);
		sweep(triangle_data, phi, closest_tri, origin, dx, -1, -1, -1);
		sweep(triangle_data, phi, closest_tri, origin, dx, +1, +1, -1);
		sweep(triangle_data, phi, closest_tri, origin, dx, -1, -1, +1);
		sweep(triangle_data, phi, closest_tri, origin, dx, +1, -1, +1);
		sweep(triangle_data, phi, closest_tri, origin, dx, -1, +1, -1);
		sweep(triangle_data, phi, closest_tri, origin, dx, +1, -1, -1);
		sweep(triangle_data, phi, closest_tri, origin, dx, -1, +1, +1);
	}
}

// figure out signs (inside/outside) from intersection counts
static void
apply_signs_from_intersection_counts(const Array3i &intersection_count,
									 Array3f &phi, int ni, int nj, int nk) {
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
