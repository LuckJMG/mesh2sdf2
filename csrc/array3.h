#ifndef ARRAY3_H
#define ARRAY3_H

#include <assert.h>
#include <stddef.h>
#include <stdlib.h>

template <class T> struct Array3 {
	int ni, nj, nk;
	T *storage;

	Array3(void) {
		ni = 0;
		nj = 0;
		nk = 0;
		storage = NULL;
	}

	Array3(int ni_, int nj_, int nk_) {
		assert(ni_ >= 0 && nj_ >= 0 && nk_ >= 0);

		ni = ni_;
		nj = nj_;
		nk = nk_;
		storage = NULL;

		size_t n = (size_t)ni_ * nj_ * nk_;
		if (n > 0) {
			storage = (T *)malloc(n * sizeof(T));
			assert(storage != NULL);
		}
	}

	Array3(int ni_, int nj_, int nk_, const T &value) {
		assert(ni_ >= 0 && nj_ >= 0 && nk_ >= 0);

		ni = ni_;
		nj = nj_;
		nk = nk_;
		storage = NULL;

		size_t n = (size_t)ni_ * nj_ * nk_;
		if (n > 0) {
			storage = (T *)malloc(n * sizeof(T));
			assert(storage != NULL);
			for (size_t i = 0; i < n; ++i) {
				storage[i] = value;
			}
		}
	}

	~Array3(void) { free(storage); }

	Array3(const Array3 &) = delete;
	Array3 &operator=(const Array3 &) = delete;

	Array3(Array3 &&o) noexcept {
		ni = o.ni;
		nj = o.nj;
		nk = o.nk;
		storage = o.storage;

		o.ni = 0;
		o.nj = 0;
		o.nk = 0;
		o.storage = NULL;
	}

	Array3 &operator=(Array3 &&o) noexcept {
		if (this != &o) {
			free(storage);

			ni = o.ni;
			nj = o.nj;
			nk = o.nk;
			storage = o.storage;

			o.ni = 0;
			o.nj = 0;
			o.nk = 0;
			o.storage = NULL;
		}

		return *this;
	}

	inline size_t index(int i, int j, int k) const {
		return (size_t)i + (size_t)ni * ((size_t)j + (size_t)nj * (size_t)k);
	}

	const T &operator()(int i, int j, int k) const {
		assert(i >= 0 && i < ni && j >= 0 && j < nj && k >= 0 && k < nk);
		// NOLINTNEXTLINE(clang-analyzer-core.uninitialized.UndefReturn)
		return storage[index(i, j, k)];
	}

	T &operator()(int i, int j, int k) {
		assert(i >= 0 && i < ni && j >= 0 && j < nj && k >= 0 && k < nk);
		// NOLINTNEXTLINE(clang-analyzer-core.uninitialized.UndefReturn)
		return storage[index(i, j, k)];
	}

	void assign(const T &value) {
		size_t n = (size_t)ni * nj * nk;
		for (size_t i = 0; i < n; ++i) {
			storage[i] = value;
		}
	}

	void resize(int ni_, int nj_, int nk_) {
		assert(ni_ >= 0 && nj_ >= 0 && nk_ >= 0);

		size_t n = (size_t)ni_ * nj_ * nk_;
		size_t on = (size_t)ni * nj * nk;
		if (n != on) {
			free(storage);
			storage = NULL;

			if (n > 0) {
				storage = (T *)malloc(n * sizeof(T));
				assert(storage != NULL);
			}
		}

		ni = ni_;
		nj = nj_;
		nk = nk_;
	}
};

typedef Array3<float> Array3f;
typedef Array3<int> Array3i;

#endif
