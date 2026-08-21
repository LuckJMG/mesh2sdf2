#ifndef UTIL_H
#define UTIL_H

#include <algorithm>

using std::max;
using std::min;

template <class T> inline T sqr(const T &x) { return x * x; }

template <class T> inline T min(T a1, T a2, T a3) {
	return min(a1, min(a2, a3));
}

template <class T> inline T max(T a1, T a2, T a3) {
	return max(a1, max(a2, a3));
}

template <class T> inline T clamp(T a, T lower, T upper) {
	if (a < lower) {
		return lower;
	}
	else if (a > upper) {
		return upper;
	}
	else {
		return a;
	}
}

#endif
