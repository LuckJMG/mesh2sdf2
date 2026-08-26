#ifndef ARRAY3_H
#define ARRAY3_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

template <class T> struct Array3 {
	int ni, nj, nk;
	std::vector<T> a;

	Array3(void) : ni(0), nj(0), nk(0) {}

	Array3(int ni_, int nj_, int nk_)
		: ni(ni_), nj(nj_), nk(nk_), a(static_cast<size_t>(ni_) * nj_ * nk_) {
		assert(ni_ >= 0 && nj_ >= 0 && nk_ >= 0);
	}

	Array3(int ni_, int nj_, int nk_, const T &value)
		: ni(ni_), nj(nj_), nk(nk_),
		  a(static_cast<size_t>(ni_) * nj_ * nk_, value) {
		assert(ni_ >= 0 && nj_ >= 0 && nk_ >= 0);
	}

	inline size_t index(int i, int j, int k) const {
		return static_cast<size_t>(i) +
			   static_cast<size_t>(ni) *
				   (static_cast<size_t>(j) +
					static_cast<size_t>(nj) * static_cast<size_t>(k));
	}

	const T &operator()(int i, int j, int k) const {
		assert(i >= 0 && i < ni && j >= 0 && j < nj && k >= 0 && k < nk);
		return a[index(i, j, k)];
	}

	T &operator()(int i, int j, int k) {
		assert(i >= 0 && i < ni && j >= 0 && j < nj && k >= 0 && k < nk);
		return a[index(i, j, k)];
	}

	void assign(const T &value) { std::fill(a.begin(), a.end(), value); }

	void resize(int ni_, int nj_, int nk_) {
		assert(ni_ >= 0 && nj_ >= 0 && nk_ >= 0);
		ni = ni_;
		nj = nj_;
		nk = nk_;
		a.resize(static_cast<size_t>(ni) * nj * nk);
	}

	void resize(int ni_, int nj_, int nk_, const T &value) {
		assert(ni_ >= 0 && nj_ >= 0 && nk_ >= 0);
		ni = ni_;
		nj = nj_;
		nk = nk_;
		a.assign(static_cast<size_t>(ni) * nj * nk, value);
	}

	size_t size(void) const { return a.size(); }
};

typedef Array3<float> Array3f;
typedef Array3<int> Array3i;

#endif
