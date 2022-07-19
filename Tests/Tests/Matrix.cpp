#include <catch2/catch.hpp>

#include <ML/Matrix.h>
#include <ML/Utils.h>

#include <vector>

TEST_CASE("Matrix Construction") {
	const ml::Matrix e;
	const ml::Matrix a = { { 1.0, 2.0 }, { 3.0, 4.0 } };
	const ml::Matrix b = a;
	const ml::Matrix c = { { 2.0, 3.0 }, { 4.0, 5.0 } };

	std::vector<ml::Matrix> vec;
	vec.push_back(a);
	vec.emplace_back(ml::Matrix(c));
	vec.push_back(b);

	REQUIRE((a == ml::Matrix({ { 1.0, 2.0 }, { 3.0, 4.0 } })));
	REQUIRE((b == ml::Matrix({ { 1.0, 2.0 }, { 3.0, 4.0 } })));
	REQUIRE((c == ml::Matrix({ { 2.0, 3.0 }, { 4.0, 5.0 } })));

	REQUIRE((vec[0] == ml::Matrix({ { 1.0, 2.0 }, { 3.0, 4.0 } })));
	REQUIRE((vec[1] == ml::Matrix({ { 2.0, 3.0 }, { 4.0, 5.0 } })));
	REQUIRE((vec[2] == ml::Matrix({ { 1.0, 2.0 }, { 3.0, 4.0 } })));
}

TEST_CASE("Matrix == Matrix") {
	const ml::Matrix a = { {1.123, 2.234}, {3.123, 4.234} };
	ml::Matrix b(2, 2);

	b[0][0] = 1.123;
	b[0][1] = 2.234;
	b[1][0] = 3.123;
	b[1][1] = 4.234;

	REQUIRE((a == b));
}

TEST_CASE("Matrix + Matrix") {
	const ml::Matrix a = { {1.0, 2.0}, {3.0, 4.0} };
	const ml::Matrix b = { {1.0, 2.0}, {3.0, 4.0} };

	const ml::Matrix c = a + b;

	REQUIRE((c == ml::Matrix({ {2.0, 4.0}, {6.0, 8.0} })));
}

TEST_CASE("Matrix - Matrix") {
	const ml::Matrix a = { {11.0, 12.0}, {13.0, 14.0} };
	const ml::Matrix b = { {1.0, 2.0}, {3.0, 4.0} };

	const ml::Matrix c = a - b;

	REQUIRE((c == ml::Matrix({ {10.0, 10.0}, {10.0, 10.0} })));
}

TEST_CASE("Matrix * Matrix") {
	const ml::Matrix a = { {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0} };
	const ml::Matrix b = { {1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0} };

	const ml::Matrix c = a * b;

	REQUIRE((c == ml::Matrix({ {11.0, 14.0, 17.0, 20.0}, {23.0, 30.0, 37.0, 44.0}, {35.0, 46.0, 57.0, 68.0} })));
}

TEST_CASE("Matrix * double") {
	const ml::Matrix a = { {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0} };
	const double b = 3.0;

	const ml::Matrix c = a * b;

	REQUIRE((c == ml::Matrix({ {3.0, 6.0}, {9.0, 12.0}, {15.0, 18.0} })));
}

TEST_CASE("double * Matrix") {
	const double a = 3.0;
	const ml::Matrix b = { {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0} };

	const ml::Matrix c = a * b;

	REQUIRE((c == ml::Matrix({ {3.0, 6.0}, {9.0, 12.0}, {15.0, 18.0} })));
}

TEST_CASE("Matrix += Matrix") {
	ml::Matrix a = { {1.0, 2.0}, {3.0, 4.0} };
	const ml::Matrix b = { {1.0, 2.0}, {3.0, 4.0} };

	a += b;

	REQUIRE((a == ml::Matrix({ {2.0, 4.0}, {6.0, 8.0} })));
}

TEST_CASE("Matrix -= Matrix") {
	ml::Matrix a = { {11.0, 12.0}, {13.0, 14.0} };
	const ml::Matrix b = { {1.0, 2.0}, {3.0, 4.0} };

	a -= b;

	REQUIRE((a == ml::Matrix({ {10.0, 10.0}, {10.0, 10.0} })));
}

TEST_CASE("Zero Matrix") {
	const ml::Matrix a = ml::Matrix::Zero(2, 3);

	REQUIRE((a == ml::Matrix({ {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0} })));
}

TEST_CASE("Transposed Matrix") {
	const ml::Matrix a = { {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0} };
	const ml::Matrix b = ml::Matrix::Transposed(a);

	REQUIRE((b == ml::Matrix({ {1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0} })));
}