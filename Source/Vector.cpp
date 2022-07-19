#include <ML/Vector.h>

#include <assert.h>
#include <memory.h>

ml::Vector::Vector()
	: m_Length(0)
	, m_Elements(nullptr) {}

ml::Vector::Vector(uint64_t length)
	: m_Length(length) {

	m_Elements = nullptr;
	if (m_Length != 0) {
		m_Elements = new double[m_Length];
	}
}

ml::Vector::Vector(std::initializer_list<double> list)
	: m_Length(list.size()) {

	m_Elements = nullptr;
	if (m_Length != 0) {
		m_Elements = new double[m_Length];

		uint64_t idx = 0;
		for (auto v : list) {
			m_Elements[idx++] = v;
		}
	}
}

ml::Vector::Vector(const Vector& other)
	: m_Length(other.m_Length) {

	m_Elements = nullptr;
	if (m_Length != 0) {
		m_Elements = new double[other.m_Length];
		memcpy(m_Elements, other.m_Elements, sizeof(double) * m_Length);
	}
}

ml::Vector::Vector(Vector&& other) noexcept
	: m_Length(other.m_Length)
	, m_Elements(other.m_Elements) {

	other.m_Elements = nullptr;
}

ml::Vector::~Vector() {
	delete [] m_Elements;
}

auto ml::Vector::operator = (const Vector& other) -> Vector& {
	m_Length = other.m_Length;

	delete [] m_Elements;
	m_Elements = nullptr;

	if (m_Length != 0) {
		m_Elements = new double[other.m_Length];
		memcpy(m_Elements, other.m_Elements, sizeof(double) * m_Length);
	}

	return *this;
}

auto ml::Vector::operator = (Vector&& other) -> Vector& {
	delete[] m_Elements;

	m_Length = other.m_Length;
	m_Elements = other.m_Elements;

	other.m_Elements = nullptr;

	return *this;
}

auto ml::Vector::operator [] (uint64_t idx) const -> double {
	assert(m_Elements != nullptr);
	assert(idx < m_Length);

	return m_Elements[idx];
}

auto ml::Vector::operator [] (uint64_t idx) -> double& {
	assert(m_Elements != nullptr);
	assert(idx < m_Length);

	return m_Elements[idx];
}

auto ml::Vector::Data() const -> const double* {
	return m_Elements;
}

auto ml::Vector::Length() const->uint64_t {
	return m_Length;
}


auto ml::Vector::Zero(uint64_t length)->Vector {
	Vector v(length);
	for (uint64_t i = 0; i < length; ++i) {
		v[i] = 0.0;
	}
	return v;
}