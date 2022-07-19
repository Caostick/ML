#include <catch2/catch.hpp>

#include <ML/Loss.h>
#include <ML/Vector.h>

TEST_CASE("CategoricalCrossentropy Loss Function") {
	const ml::LossFunc_CategoricalCrossentropy func;

	const ml::Vector a = { 0.03511902695934, 0.25949646034242, 0.70538451269824 };
	const ml::Vector y = { 0.0, 1.0, 0.0 };

	const double e = func.F(a, y);
	const double properValue = 1.3490122167681828;

	REQUIRE(fabs(e - properValue) < 0.0001);
}

TEST_CASE("BinaryCrossentropy Loss Function") {
	const ml::LossFunc_BinaryCrossentropy func;

	const ml::Vector a = { 0.03511902695934, 0.25949646034242, 0.70538451269824 };
	const ml::Vector y = { 0.0, 1.0, 0.0 };

	const double e = func.F(a, y);
	const double properValue = 0.029838585475254271;

	REQUIRE(fabs(e - properValue) < 0.0001);
}

TEST_CASE("SquaredHinge Loss Function") {
	const ml::LossFunc_SquaredHinge func;

	const ml::Vector a = { 0.03511902695934, 0.25949646034242, 0.70538451269824 };
	const ml::Vector y = { 0.0, 1.0, 0.0 };

	const double e = func.F(a, y);
	const double properValue = 0.84944849741513506;

	REQUIRE(fabs(e - properValue) < 0.0001);
}

TEST_CASE("MeanSquaredError Loss Function") {
	const ml::LossFunc_MeanSquaredError func;

	const ml::Vector a = { 0.03511902695934, 0.25949646034242, 0.70538451269824 };
	const ml::Vector y = { 0.2, 0.7, 0.3 };

	const double e = func.F(a, y);
	const double properValue = 0.12852190228576049;

	REQUIRE(fabs(e - properValue) < 0.0001);
}

TEST_CASE("MeanAbsoluteError Loss Function") {
	const ml::LossFunc_MeanAbsoluteError func;

	const ml::Vector a = { 0.03511902695934, 0.25949646034242, 0.70538451269824 };
	const ml::Vector y = { 0.2, 0.7, 0.3 };

	const double e = func.F(a, y);
	const double properValue = 0.33692300846549333;

	REQUIRE(fabs(e - properValue) < 0.0001);
}

TEST_CASE("MeanSquaredLogarithmicError Loss Function") {
	const ml::LossFunc_MeanSquaredLogarithmicError func;

	const ml::Vector a = { 0.03511902695934, 0.25949646034242, 0.70538451269824 };
	const ml::Vector y = { 0.2, 0.7, 0.3 };

	const double e = func.F(a, y);
	const double properValue = 0.061822790171077814;

	REQUIRE(fabs(e - properValue) < 0.0001);
}

TEST_CASE("Poisson Loss Function") {
	const ml::LossFunc_Poisson func;

	const ml::Vector a = { 0.03511902695934, 0.25949646034242, 0.70538451269824 };
	const ml::Vector y = { 0.2, 0.7, 0.3 };

	const double e = func.F(a, y);
	const double properValue = 0.90627155337394016;

	REQUIRE(fabs(e - properValue) < 0.0001);
}