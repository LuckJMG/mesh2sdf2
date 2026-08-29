#ifndef VEC_H
#define VEC_H

#include <assert.h>
#include <math.h>

struct Vec3f {
	float x, y, z;
	Vec3f(void) {}
	Vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
	float &operator[](int i) {
		assert(0 <= i && i < 3);
		return (&x)[i];
	}
	const float &operator[](int i) const {
		assert(0 <= i && i < 3);
		return (&x)[i];
	}
};

struct Vec3ui {
	unsigned int x, y, z;
	Vec3ui(void) {}
	Vec3ui(unsigned int x_, unsigned int y_, unsigned int z_)
		: x(x_), y(y_), z(z_) {}
	unsigned int &operator[](int i) {
		assert(0 <= i && i < 3);
		return (&x)[i];
	}
	const unsigned int &operator[](int i) const {
		assert(0 <= i && i < 3);
		return (&x)[i];
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

inline float dist2(const Vec3f &a, const Vec3f &b) {
	float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
	return dx * dx + dy * dy + dz * dz;
}

inline float dist(const Vec3f &a, const Vec3f &b) { return sqrt(dist2(a, b)); }

#endif
