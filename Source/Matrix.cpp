#include <ML/Matrix.h>

#include <assert.h>
#include <memory.h>

ml::Matrix::Matrix()
	: m_Elements(nullptr)
	, m_Cols(0)
	, m_Rows(0) {}

ml::Matrix::Matrix(uint64_t cols, uint64_t rows)
	: m_Cols(cols)
	, m_Rows(rows) {
	assert(m_Cols * m_Rows != 0);

	m_Elements = nullptr;
	if (m_Cols * m_Rows != 0) {
		m_Elements = new double[m_Cols * m_Rows];
	}
}

ml::Matrix::Matrix(std::initializer_list<std::initializer_list<double>> list) {
	m_Rows = list.size();
	m_Cols = list.begin()->size();

	m_Elements = new double[m_Cols * m_Rows];

	uint64_t i = 0;
	for (const auto& li : list) {
		assert(li.size() == m_Cols);

		uint64_t j = 0;
		for (const auto& lj : li) {
			m_Elements[i * m_Cols + j] = lj;
			j++;
		}
		i++;
	}
}

ml::Matrix::Matrix(const Matrix& other)
	: m_Cols(other.m_Cols)
	, m_Rows(other.m_Rows) {

	m_Elements = nullptr;
	if (m_Cols * m_Rows != 0) {
		m_Elements = new double[m_Cols * m_Rows];
		memcpy(m_Elements, other.m_Elements, sizeof(double) * m_Cols * m_Rows);
	}
}

ml::Matrix::Matrix(Matrix&& other)
	: m_Elements(other.m_Elements)
	, m_Cols(other.m_Cols)
	, m_Rows(other.m_Rows) {
	other.m_Elements = nullptr;
}

ml::Matrix::~Matrix() {
	delete [] m_Elements;
}

auto ml::Matrix::operator = (const Matrix& other)->Matrix& {
	m_Cols = other.m_Cols;
	m_Rows = other.m_Rows;

	delete [] m_Elements;
	m_Elements = nullptr;
	if (m_Cols * m_Rows != 0) {
		m_Elements = new double[m_Cols * m_Rows];
		memcpy(m_Elements, other.m_Elements, sizeof(double) * m_Cols * m_Rows);
	}

	return *this;
}

auto ml::Matrix::operator = (Matrix&& other) -> Matrix& {
	m_Elements = other.m_Elements;
	m_Cols = other.m_Cols;
	m_Rows = other.m_Rows;
	other.m_Elements = nullptr;

	return *this;
}

auto ml::Matrix::operator [] (uint64_t idx) const -> const double* {
	assert(m_Elements != nullptr);
	assert(idx < m_Rows);

	return m_Elements + idx * m_Cols;
}

auto ml::Matrix::operator [] (uint64_t idx)->double* {
	assert(m_Elements != nullptr);
	assert(idx < m_Rows);

	return m_Elements + idx * m_Cols;
}

auto ml::Matrix::Cols() const->uint64_t {
	return m_Cols;
}

auto ml::Matrix::Rows() const->uint64_t {
	return m_Rows;
}




auto ml::Matrix::Zero(uint64_t cols, uint64_t rows)->Matrix {
	Matrix m(cols, rows);

	for (uint64_t i = 0; i < rows; ++i) {
		for (uint64_t j = 0; j < cols; ++j) {
			m[i][j] = 0.0;
		}
	}

	return m;
}

auto ml::Matrix::Transposed(const Matrix mat)->Matrix {
	Matrix m(mat.Rows(), mat.Cols());

	for (uint64_t i = 0; i < m.Cols(); ++i) {
		for (uint64_t j = 0; j < m.Rows(); ++j) {
			m[j][i] = mat[i][j];
		}
	}

	return m;
}