#pragma once

#include <ML/Vector.h>

namespace ml {
	class LossFunc {
	public:
		virtual auto F(const Vector& predicted, const Vector& expected) const -> double = 0;
	};
}

namespace ml {
	class LossFunc_CategoricalCrossentropy : public LossFunc {
	public:
		auto F(const Vector& predicted, const Vector& expected) const -> double override;
	};

	class LossFunc_BinaryCrossentropy : public LossFunc {
	public:
		auto F(const Vector& predicted, const Vector& expected) const -> double override;
	};

	class LossFunc_SquaredHinge : public LossFunc {
	public:
		auto F(const Vector& predicted, const Vector& expected) const -> double override;
	};

	class LossFunc_MeanSquaredError : public LossFunc {
	public:
		auto F(const Vector& predicted, const Vector& expected) const -> double override;
	};

	class LossFunc_MeanAbsoluteError : public LossFunc {
	public:
		auto F(const Vector& predicted, const Vector& expected) const -> double override;
	};

	class LossFunc_MeanSquaredLogarithmicError : public LossFunc {
	public:
		auto F(const Vector& predicted, const Vector& expected) const -> double override;
	};

	class LossFunc_Poisson : public LossFunc {
	public:
		auto F(const Vector& predicted, const Vector& expected) const -> double override;
	};
}