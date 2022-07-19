#include <ML/Utils.h>
#include <ML/Vector.h>
#include <ML/Matrix.h>

#include <assert.h>
#include <stdlib.h>
#include <math.h>

auto RndRange(double min, double max) -> double {
	return ((rand() % RAND_MAX) / double(RAND_MAX)) * (max - min) + min;
}

bool operator == (const ml::Vector& a, const ml::Vector& b) {
	assert(a.Length() == b.Length());

	for (uint64_t i = 0; i < a.Length(); ++i) {
		if (fabs(a[i] - b[i]) > 0.000001) {
			return false;
		}
	}

	return true;
}

auto operator + (const ml::Vector& a, const ml::Vector& b)->ml::Vector {
	assert(a.Length() == b.Length());

	ml::Vector v(a.Length());

	for (uint64_t i = 0; i < a.Length(); ++i) {
		v[i] = a[i] + b[i];
	}

	return v;
}

auto operator - (const ml::Vector& a, const ml::Vector& b)->ml::Vector {
	assert(a.Length() == b.Length());

	ml::Vector v(a.Length());

	for (uint64_t i = 0; i < a.Length(); ++i) {
		v[i] = a[i] - b[i];
	}

	return v;
}

auto operator * (const ml::Vector& a, const ml::Vector& b)->ml::Vector {
	assert(a.Length() == b.Length());

	ml::Vector v(a.Length());

	for (uint64_t i = 0; i < a.Length(); ++i) {
		v[i] = a[i] * b[i];
	}

	return v;
}

auto operator * (const ml::Vector& a, double b)->ml::Vector {
	ml::Vector v(a.Length());

	for (uint64_t i = 0; i < a.Length(); ++i) {
		v[i] = a[i] * b;
	}

	return v;
}

auto operator * (double a, const ml::Vector& b)->ml::Vector {
	ml::Vector v(b.Length());

	for (uint64_t i = 0; i < b.Length(); ++i) {
		v[i] = a * b[i];
	}

	return v;
}

auto operator += (ml::Vector& a, const ml::Vector& b)-> const ml::Vector& {
	assert(a.Length() == b.Length());

	for (uint64_t i = 0; i < a.Length(); ++i) {
		a[i] += b[i];
	}

	return a;
}

auto operator -= (ml::Vector& a, const ml::Vector& b)-> const ml::Vector& {
	assert(a.Length() == b.Length());

	for (uint64_t i = 0; i < a.Length(); ++i) {
		a[i] -= b[i];
	}

	return a;
}

auto operator *= (ml::Vector& a, const ml::Vector& b)-> const ml::Vector& {
	assert(a.Length() == b.Length());

	for (uint64_t i = 0; i < a.Length(); ++i) {
		a[i] *= b[i];
	}

	return a;
}


bool operator == (const ml::Matrix& a, const ml::Matrix& b) {
	assert(a.Cols() == b.Cols());
	assert(a.Rows() == b.Rows());

	for (uint64_t i = 0; i < a.Rows(); ++i) {
		for (uint64_t j = 0; j < a.Cols(); ++j) {
			if (fabs(a[i][j] - b[i][j]) > 0.000001) {
				return false;
			}
		}
	}

	return true;
}

auto operator + (const ml::Matrix& a, const ml::Matrix& b)->ml::Matrix {
	assert(a.Cols() == b.Cols());
	assert(a.Rows() == b.Rows());

	ml::Matrix m(a.Cols(), a.Rows());

	for (uint64_t i = 0; i < a.Rows(); ++i) {
		for (uint64_t j = 0; j < a.Cols(); ++j) {
			m[i][j] = a[i][j] + b[i][j];
		}
	}

	return m;
}

auto operator - (const ml::Matrix& a, const ml::Matrix& b)->ml::Matrix {
	assert(a.Cols() == b.Cols());
	assert(a.Rows() == b.Rows());

	ml::Matrix m(a.Cols(), a.Rows());

	for (uint64_t i = 0; i < a.Rows(); ++i) {
		for (uint64_t j = 0; j < a.Cols(); ++j) {
			m[i][j] = a[i][j] - b[i][j];
		}
	}

	return m;
}

auto operator * (const ml::Matrix& a, const ml::Matrix& b)->ml::Matrix {
	assert(a.Cols() == b.Rows());

	ml::Matrix m(b.Cols(), a.Rows());

	for (uint64_t i = 0; i < m.Rows(); ++i) {
		for (uint64_t j = 0; j < m.Cols(); ++j) {
			m[i][j] = 0.0;

			for (uint64_t k = 0; k < a.Cols(); ++k) {
				m[i][j] += a[i][k] * b[k][j];
			}
		}
	}

	return m;
}

auto operator * (const ml::Matrix& a, double b)->ml::Matrix {
	ml::Matrix m(a.Cols(), a.Rows());

	for (uint64_t i = 0; i < m.Rows(); ++i) {
		for (uint64_t j = 0; j < m.Cols(); ++j) {
			m[i][j] = a[i][j] * b;
		}
	}

	return m;
}

auto operator * (double a, const ml::Matrix& b)->ml::Matrix {
	ml::Matrix m(b.Cols(), b.Rows());

	for (uint64_t i = 0; i < m.Rows(); ++i) {
		for (uint64_t j = 0; j < m.Cols(); ++j) {
			m[i][j] = a * b[i][j];
		}
	}

	return m;
}

auto operator += (ml::Matrix& a, const ml::Matrix& b)-> const ml::Matrix& {
	assert(a.Cols() == b.Cols());
	assert(a.Rows() == b.Rows());

	for (uint64_t i = 0; i < a.Rows(); ++i) {
		for (uint64_t j = 0; j < a.Cols(); ++j) {
			a[i][j] += b[i][j];
		}
	}

	return a;
}

auto operator -= (ml::Matrix& a, const ml::Matrix& b)-> const ml::Matrix& {
	assert(a.Cols() == b.Cols());
	assert(a.Rows() == b.Rows());

	for (uint64_t i = 0; i < a.Rows(); ++i) {
		for (uint64_t j = 0; j < a.Cols(); ++j) {
			a[i][j] -= b[i][j];
		}
	}

	return a;
}




auto operator * (const ml::Matrix& a, const ml::Vector& b)->ml::Vector {
	assert(a.Cols() == b.Length());

	ml::Vector v(a.Rows());

	for (uint64_t i = 0; i < a.Rows(); ++i) {
		v[i] = 0.0;
		for (uint64_t j = 0; j < a.Cols(); ++j) {
			v[i] += a[i][j] * b[j];
		}
	}

	return v;
}

auto operator * (const ml::Vector& a, const ml::Matrix& b)->ml::Vector {
	assert(a.Length() == b.Rows());

	ml::Vector v(b.Cols());

	for (uint64_t i = 0; i < b.Cols(); ++i) {
		v[i] = 0.0;
		for (uint64_t j = 0; j < b.Rows(); ++j) {
			v[i] += a[j] * b[j][i];
		}
	}

	return v;
}

auto operator & (const ml::Vector& a, const ml::Vector& b)->ml::Matrix {
	ml::Matrix m(b.Length(), a.Length());

	for (uint64_t i = 0; i < m.Rows(); ++i) {
		for (uint64_t j = 0; j < m.Cols(); ++j) {
			m[i][j] = a[i] * b[j];
		}
	}

	return m;
}

auto operator ^ (const ml::Vector& a, const ml::Vector& b)->double {
	assert(a.Length() == b.Length());

	double dp = 0.0;
	for (uint64_t i = 0; i < a.Length(); ++i) {
		dp += a[i] * b[i];
	}
	return dp;
}