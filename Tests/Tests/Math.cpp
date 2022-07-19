#include <catch2/catch.hpp>

#include <ML/Vector.h>
#include <ML/Matrix.h>
#include <ML/Utils.h>

TEST_CASE("Matrix * Vector") {
	const ml::Matrix a = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
	const ml::Vector b = {1.0, 2.0, 3.0};

	const ml::Vector c = a * b;

	REQUIRE((c == ml::Vector({14.0, 32.0})));
}

TEST_CASE("Vector * Matrix") {
	const ml::Vector a = { 1.0, 2.0, 3.0 };
	const ml::Matrix b = { {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0} };

	const ml::Vector c = a * b;

	REQUIRE((c == ml::Vector({ 22.0, 28.0 })));
}

TEST_CASE("Vector & Vector") {
	const ml::Vector a = { 1.0, 2.0 };
	const ml::Vector b = { 1.0, 2.0, 3.0 };

	const ml::Matrix c = a & b;

	REQUIRE((c == ml::Matrix({ { 1.0, 2.0, 3.0 }, { 2.0, 4.0, 6.0 } })));
}

TEST_CASE("Vector ^ Vector") {
	const ml::Vector a = { 2.0, 3.0, 4.0 };
	const ml::Vector b = { 1.0, 2.0, 3.0 };

	const double c = a ^ b;

	REQUIRE(c == 20.0);
}