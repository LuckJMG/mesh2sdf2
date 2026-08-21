#ifndef VEC_H
#define VEC_H

#include "util.h"
#include <cassert>
#include <cmath>

// Thin wrapper around fixed-size C-style arrays, trimmed from upstream
// SDFGen vec.h to the surface used by makelevelset3.cpp:
//   Vec<3,float>        -> Vec3f
//   Vec<3,unsigned int> -> Vec3ui

template <unsigned int N, class T> struct Vec {
	T v[N];

	Vec<N, T>(void) {}

	Vec<N, T>(T v0, T v1, T v2) {
		assert(N == 3);
		v[0] = v0;
		v[1] = v1;
		v[2] = v2;
	}

	T &operator[](int index) {
		assert(0 <= index && (unsigned int)index < N);
		return v[index];
	}

	const T &operator[](int index) const {
		assert(0 <= index && (unsigned int)index < N);
		return v[index];
	}

	Vec<N, T> operator+=(const Vec<N, T> &w) {
		for (unsigned int i = 0; i < N; ++i) {
			v[i] += w[i];
		}
		return *this;
	}

	Vec<N, T> operator+(const Vec<N, T> &w) const {
		Vec<N, T> sum(*this);
		sum += w;
		return sum;
	}

	Vec<N, T> operator-=(const Vec<N, T> &w) {
		for (unsigned int i = 0; i < N; ++i) {
			v[i] -= w[i];
		}
		return *this;
	}

	Vec<N, T> operator-(const Vec<N, T> &w) const // binary subtraction
	{
		Vec<N, T> diff(*this);
		diff -= w;
		return diff;
	}

	Vec<N, T> operator*=(T a) {
		for (unsigned int i = 0; i < N; ++i) {
			v[i] *= a;
		}
		return *this;
	}
};

typedef Vec<3, float> Vec3f;
typedef Vec<3, unsigned int> Vec3ui;

template <unsigned int N, class T> T mag2(const Vec<N, T> &a) {
	T l = sqr(a.v[0]);
	for (unsigned int i = 1; i < N; ++i) {
		l += sqr(a.v[i]);
	}
	return l;
}

template <unsigned int N, class T>
inline T dist2(const Vec<N, T> &a, const Vec<N, T> &b) {
	T d = sqr(a.v[0] - b.v[0]);
	for (unsigned int i = 1; i < N; ++i) {
		d += sqr(a.v[i] - b.v[i]);
	}
	return d;
}

template <unsigned int N, class T>
inline T dist(const Vec<N, T> &a, const Vec<N, T> &b) {
	return std::sqrt(dist2(a, b));
}

template <unsigned int N, class T>
inline T dot(const Vec<N, T> &a, const Vec<N, T> &b) {
	T d = a.v[0] * b.v[0];
	for (unsigned int i = 1; i < N; ++i) {
		d += a.v[i] * b.v[i];
	}
	return d;
}

template <unsigned int N, class T>
inline Vec<N, T> operator*(T a, const Vec<N, T> &v) {
	Vec<N, T> w(v);
	w *= a;
	return w;
}

template <unsigned int N, class T>
inline void assign(const Vec<N, T> &a, T &a0, T &a1, T &a2) {
	assert(N == 3);
	a0 = a.v[0];
	a1 = a.v[1];
	a2 = a.v[2];
}

#endif
