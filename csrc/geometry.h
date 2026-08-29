#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <assert.h>
#include <math.h>

#include "vec.h"

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

static inline int clamp_int(int v, int lo, int hi) {
	if (v < lo) {
		return lo;
	}
	if (v > hi) {
		return hi;
	}
	return v;
}

static inline float min_f(float a, float b) { return a < b ? a : b; }

static inline double min3(double a, double b, double c) {
	double m = a;
	if (b < m) {
		m = b;
	}
	if (c < m) {
		m = c;
	}
	return m;
}

static inline double max3(double a, double b, double c) {
	double m = a;
	if (b > m) {
		m = b;
	}
	if (c > m) {
		m = c;
	}
	return m;
}

// find distance x0 is from segment x1-x2
static inline float point_segment_distance(const Vec3f &x0, const Vec3f &x1,
										   const Vec3f &x2,
										   const SegmentData &edge) {
	float s12 = (float)(dot(x2 - x0, edge.delta) / edge.squared_length);
	if (s12 < 0) {
		s12 = 0;
	}
	else if (s12 > 1) {
		s12 = 1;
	}
	return dist(x0, s12 * x1 + (1 - s12) * x2);
}

// find distance x0 is from triangle x1-x2-x3
static inline float point_triangle_distance(const Vec3f &x0,
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
	// outside triangle: clamp to the two edges of the dominant barycentric
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

// calculate twice signed area of triangle (0,0)-(x1,y1)-(x2,y2)
// return an SOS-determined sign (-1, +1, or 0 only if it's a truly degenerate
// triangle)
static inline int orientation(double x1, double y1, double x2, double y2,
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
static inline bool point_in_triangle_2d(double x0, double y0, double x1,
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
	assert(sum != 0); // if the SOS signs match and are nonkero, there's no way
					  // all of a, b, and c are zero.

	a /= sum;
	b /= sum;
	c /= sum;

	return true;
}

// per-triangle precomputed data plus the triangle vertices in grid
// coordinates to high precision, shared by both init paths
struct TriSetup {
	TriangleData triangle;
	double p_i, p_j, p_k;
	double q_i, q_j, q_k;
	double r_i, r_j, r_k;
};

static inline TriSetup setup_triangle(const Vec3ui &t, const Vec3f *x,
									  const Vec3f &origin, float dx) {
	unsigned int p = t.x, q = t.y, r = t.z;
	TriSetup s;

	TriangleData &triangle = s.triangle;
	triangle.x1 = x[p];
	triangle.x2 = x[q];
	triangle.x3 = x[r];
	triangle.x13 = triangle.x1 - triangle.x3;
	triangle.x23 = triangle.x2 - triangle.x3;
	triangle.m13 = mag2(triangle.x13);
	triangle.m23 = mag2(triangle.x23);
	triangle.dot = dot(triangle.x13, triangle.x23);

	double det = (double)triangle.m13 * triangle.m23 -
				 (double)triangle.dot * triangle.dot;
	double det_clamped = det > 1e-30 ? det : 1e-30;

	triangle.invdet = (float)(1.0 / det_clamped);
	triangle.edge12 = {triangle.x2 - triangle.x1,
					   (double)mag2(triangle.x2 - triangle.x1)};
	triangle.edge13 = {triangle.x3 - triangle.x1,
					   (double)mag2(triangle.x3 - triangle.x1)};
	triangle.edge23 = {triangle.x3 - triangle.x2,
					   (double)mag2(triangle.x3 - triangle.x2)};

	s.p_i = ((double)x[p].x - origin.x) / dx;
	s.p_j = ((double)x[p].y - origin.y) / dx;
	s.p_k = ((double)x[p].z - origin.z) / dx;
	s.q_i = ((double)x[q].x - origin.x) / dx;
	s.q_j = ((double)x[q].y - origin.y) / dx;
	s.q_k = ((double)x[q].z - origin.z) / dx;
	s.r_i = ((double)x[r].x - origin.x) / dx;
	s.r_j = ((double)x[r].y - origin.y) / dx;
	s.r_k = ((double)x[r].z - origin.z) / dx;
	return s;
}

#endif
