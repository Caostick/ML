#include <catch2/catch.hpp>

#include <ML/Vector.h>
#include <ML/Utils.h>

#include <vector>

TEST_CASE("Vector Construction") {
	const ml::Vector a = { 1.0, 2.0 };
	const ml::Vector b = a;
	const ml::Vector c = { 2.0, 3.0 };

	std::vector<ml::Vector> vec;
	vec.push_back(a);
	vec.emplace_back(ml::Vector(c));
	vec.push_back(b);

	REQUIRE((a == ml::Vector({1.0, 2.0})));
	REQUIRE((b == ml::Vector({ 1.0, 2.0 })));
	REQUIRE((c == ml::Vector({ 2.0, 3.0 })));

	REQUIRE((vec[0] == ml::Vector({ 1.0, 2.0 })));
	REQUIRE((vec[1] == ml::Vector({ 2.0, 3.0 })));
	REQUIRE((vec[2] == ml::Vector({ 1.0, 2.0 })));
}

TEST_CASE("Vector == Vector") {
	const ml::Vector a = { 1.123, 2.234 };
	ml::Vector b(2);

	b[0] = 1.123;
	b[1] = 2.234;

	REQUIRE((a == b));
}

TEST_CASE("Vector + Vector") {
	const ml::Vector a = {1.0, 2.0};
	const ml::Vector b = {1.0, 2.0};

	const ml::Vector c = a + b;
	REQUIRE((c == ml::Vector({ 2.0, 4.0 })));
}

TEST_CASE("Vector - Vector") {
	const ml::Vector a = { 6.0, 7.0 };
	const ml::Vector b = { 1.0, 2.0 };

	const ml::Vector c = a - b;
	REQUIRE((c == ml::Vector({ 5.0, 5.0 })));
}

TEST_CASE("Vector * Vector") {
	const ml::Vector a = { 1.0, 2.0, 3.0 };
	const ml::Vector b = { 4.0, 5.0, 6.0 };

	const ml::Vector c = a * b;
	REQUIRE((c == ml::Vector({ 4.0, 10.0, 18.0 })));
}

TEST_CASE("Vector * double") {
	const ml::Vector a = { 1.0, 2.0, 3.0 };
	const double b = 3.0;

	const ml::Vector c = a * b;
	REQUIRE((c == ml::Vector({ 3.0, 6.0, 9.0 })));
}

TEST_CASE("double * Vector") {
	const double a = 3.0;
	const ml::Vector b = { 1.0, 2.0, 3.0 };

	const ml::Vector c = a * b;
	REQUIRE((c == ml::Vector({ 3.0, 6.0, 9.0 })));
}

TEST_CASE("Vector += Vector") {
	ml::Vector a = { 1.0, 2.0 };
	const ml::Vector b = { 1.0, 2.0 };

	a += b;
	REQUIRE((a == ml::Vector({ 2.0, 4.0 })));
}

TEST_CASE("Vector -= Vector") {
	ml::Vector a = { 6.0, 7.0 };
	const ml::Vector b = { 1.0, 2.0 };

	a -= b;
	REQUIRE((a == ml::Vector({ 5.0, 5.0 })));
}

TEST_CASE("Vector *= Vector") {
	ml::Vector a = { 1.0, 2.0, 3.0 };
	const ml::Vector b = { 4.0, 5.0, 6.0 };

	a *= b;
	REQUIRE((a == ml::Vector({ 4.0, 10.0, 18.0 })));
}