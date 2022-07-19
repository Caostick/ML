#pragma once

#include <ML/Vector.h>

#include <stdint.h>
#include <initializer_list>

namespace ml {
	class Matrix {
	public:

		Matrix();
		Matrix(uint64_t cols, uint64_t rows);
		Matrix(std::initializer_list<std::initializer_list<double>> list);
		Matrix(const Matrix& other);
		Matrix(Matrix&& other);

		~Matrix();

		auto operator = (const Matrix& other)->Matrix&;
		auto operator = (Matrix&& other) -> Matrix&;

		auto operator [] (uint64_t idx) const->const double*;
		auto operator [] (uint64_t idx)->double*;

		auto Cols() const->uint64_t;
		auto Rows() const->uint64_t;

		static auto Zero(uint64_t cols, uint64_t rows)->Matrix;
		static auto Transposed(const Matrix mat) ->Matrix;

	private:
		double* m_Elements;
		uint64_t m_Cols;
		uint64_t m_Rows;
	};
}