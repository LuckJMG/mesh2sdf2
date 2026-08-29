#ifndef ARRAY3_H
#define ARRAY3_H

#include <assert.h>
#include <stddef.h>
#include <stdlib.h>

template <class T> struct Array3 {
	int ni, nj, nk;
	T *a;

	Array3(void) : ni(0), nj(0), nk(0), a(NULL) {}

	Array3(int ni_, int nj_, int nk_) : ni(ni_), nj(nj_), nk(nk_), a(NULL) {
		assert(ni_ >= 0 && nj_ >= 0 && nk_ >= 0);
		size_t n = (size_t)ni_ * nj_ * nk_;
		if (n > 0) {
			a = (T *)malloc(n * sizeof(T));
			assert(a != NULL);
		}
	}

	Array3(int ni_, int nj_, int nk_, const T &value)
		: ni(ni_), nj(nj_), nk(nk_), a(NULL) {
		assert(ni_ >= 0 && nj_ >= 0 && nk_ >= 0);
		size_t n = (size_t)ni_ * nj_ * nk_;
		if (n > 0) {
			a = (T *)malloc(n * sizeof(T));
			assert(a != NULL);
			for (size_t i = 0; i < n; ++i) {
				a[i] = value;
			}
		}
	}

	~Array3(void) { free(a); }

	Array3(const Array3 &) = delete;
	Array3 &operator=(const Array3 &) = delete;

	Array3(Array3 &&o) noexcept : ni(o.ni), nj(o.nj), nk(o.nk), a(o.a) {
		o.ni = 0;
		o.nj = 0;
		o.nk = 0;
		o.a = NULL;
	}

	Array3 &operator=(Array3 &&o) noexcept {
		if (this != &o) {
			free(a);
			ni = o.ni;
			nj = o.nj;
			nk = o.nk;
			a = o.a;
			o.ni = 0;
			o.nj = 0;
			o.nk = 0;
			o.a = NULL;
		}
		return *this;
	}

	inline size_t index(int i, int j, int k) const {
		return (size_t)i + (size_t)ni * ((size_t)j + (size_t)nj * (size_t)k);
	}

	const T &operator()(int i, int j, int k) const {
		assert(i >= 0 && i < ni && j >= 0 && j < nj && k >= 0 && k < nk);
		// NOLINTNEXTLINE(clang-analyzer-core.uninitialized.UndefReturn)
		return a[index(i, j, k)];
	}

	T &operator()(int i, int j, int k) {
		assert(i >= 0 && i < ni && j >= 0 && j < nj && k >= 0 && k < nk);
		// NOLINTNEXTLINE(clang-analyzer-core.uninitialized.UndefReturn)
		return a[index(i, j, k)];
	}

	T *data(void) { return a; }
	const T *data(void) const { return a; }

	void assign(const T &value) {
		size_t n = (size_t)ni * nj * nk;
		for (size_t i = 0; i < n; ++i) {
			a[i] = value;
		}
	}

	void resize(int ni_, int nj_, int nk_) {
		assert(ni_ >= 0 && nj_ >= 0 && nk_ >= 0);
		size_t n = (size_t)ni_ * nj_ * nk_;
		size_t on = (size_t)ni * nj * nk;
		if (n != on) {
			free(a);
			a = NULL;
			if (n > 0) {
				a = (T *)malloc(n * sizeof(T));
				assert(a != NULL);
			}
		}
		ni = ni_;
		nj = nj_;
		nk = nk_;
	}

	void resize(int ni_, int nj_, int nk_, const T &value) {
		assert(ni_ >= 0 && nj_ >= 0 && nk_ >= 0);
		free(a);
		a = NULL;
		ni = ni_;
		nj = nj_;
		nk = nk_;
		size_t n = (size_t)ni * nj * nk;
		if (n > 0) {
			a = (T *)malloc(n * sizeof(T));
			assert(a != NULL);
			for (size_t i = 0; i < n; ++i) {
				a[i] = value;
			}
		}
	}

	size_t size(void) const { return (size_t)ni * nj * nk; }
};

typedef Array3<float> Array3f;
typedef Array3<int> Array3i;

#endif
