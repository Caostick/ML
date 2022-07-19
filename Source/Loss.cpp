#include <ML/Loss.h>

#include <assert.h>
#include <math.h>

namespace {
	auto max(double a, double b) -> double {
		return (a >= b) ? a : b;
	}
}

auto ml::LossFunc_CategoricalCrossentropy::F(const Vector& predicted, const Vector& expected) const -> double {
	assert(predicted.Length() == expected.Length());

	double sum = 0.0;
	for (uint64_t i = 0; i < predicted.Length(); ++i) {
		sum += expected[i] * log(predicted[i]);
	}
	
	return -sum;
}

auto ml::LossFunc_BinaryCrossentropy::F(const Vector& predicted, const Vector& expected) const -> double {
	assert(predicted.Length() == expected.Length());

	double sum = 0.0;
	for (uint64_t i = 0; i < predicted.Length(); ++i) {
		sum += expected[i] * log(predicted[i]) + (1.0 - expected[i]) * (1.0 - predicted[i]);
	}

	return -sum / predicted.Length();
}

auto ml::LossFunc_SquaredHinge::F(const Vector& predicted, const Vector& expected) const -> double {
	assert(predicted.Length() == expected.Length());

	double sum = 0.0;
	for (uint64_t i = 0; i < predicted.Length(); ++i) {
		sum += pow(max(0.0, 1.0 - predicted[i] * expected[i]), 2.0);
	}

	return sum / predicted.Length();
}

auto ml::LossFunc_MeanSquaredError::F(const Vector& predicted, const Vector& expected) const -> double {
	assert(predicted.Length() == expected.Length());

	double sum = 0.0;
	for (uint64_t i = 0; i < predicted.Length(); ++i) {
		sum += pow(expected[i] - predicted[i], 2.0);
	}

	return sum / predicted.Length();
}

auto ml::LossFunc_MeanAbsoluteError::F(const Vector& predicted, const Vector& expected) const -> double {
	assert(predicted.Length() == expected.Length());

	double sum = 0.0;
	for (uint64_t i = 0; i < predicted.Length(); ++i) {
		sum += fabs(expected[i] - predicted[i]);
	}

	return sum / predicted.Length();
}

auto ml::LossFunc_MeanSquaredLogarithmicError::F(const Vector& predicted, const Vector& expected) const -> double {
	assert(predicted.Length() == expected.Length());

	double sum = 0.0;
	for (uint64_t i = 0; i < predicted.Length(); ++i) {
		sum += pow(log(expected[i] + 1.0) - log(predicted[i] + 1.0), 2.0);
	}

	return sum / predicted.Length();
}

auto ml::LossFunc_Poisson::F(const Vector& predicted, const Vector& expected) const -> double {
	assert(predicted.Length() == expected.Length());

	double sum = 0.0;
	for (uint64_t i = 0; i < predicted.Length(); ++i) {
		sum += predicted[i] - expected[i] * log(predicted[i]);
	}

	return sum / predicted.Length();
}