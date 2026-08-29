#ifndef VEC_H
#define VEC_H

#include <assert.h>
#include <math.h>

struct Vec3f {
	float x, y, z;
	Vec3f(void) {}
	Vec3f(float x_, float y_, float z_) {
		x = x_;
		y = y_;
		z = z_;
	}
};

struct Vec3ui {
	unsigned int x, y, z;
	Vec3ui(void) {}
	Vec3ui(unsigned int x_, unsigned int y_, unsigned int z_) {
		x = x_;
		y = y_;
		z = z_;
	}
};

inline Vec3f operator+(const Vec3f &a, const Vec3f &b) {
	return Vec3f(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline Vec3f operator-(const Vec3f &a, const Vec3f &b) {
	return Vec3f(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline Vec3f operator*(float s, const Vec3f &v) {
	return Vec3f(s * v.x, s * v.y, s * v.z);
}

inline float dot(const Vec3f &a, const Vec3f &b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float mag2(const Vec3f &a) { return dot(a, a); }

inline float dist(const Vec3f &a, const Vec3f &b) {
	float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
	return sqrt(dx * dx + dy * dy + dz * dz);
}

#endif
