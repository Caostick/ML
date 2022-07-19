#pragma once

#include <stdint.h>
#include <initializer_list>

namespace ml {
	class Vector {
	public:
		Vector();
		Vector(uint64_t length);
		Vector(std::initializer_list<double> list);
		Vector(const Vector& other);
		Vector(Vector&& other) noexcept;

		~Vector();

		auto operator = (const Vector& other) -> Vector&;
		auto operator = (Vector&& other) -> Vector&;

		auto operator [] (uint64_t idx) const -> double;
		auto operator [] (uint64_t idx) -> double&;

		auto Data() const -> const double*;
		auto Length() const->uint64_t;

		static auto Zero(uint64_t length)->Vector;

	private:
		double* m_Elements;
		uint64_t m_Length;
	};
}