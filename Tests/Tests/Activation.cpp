#include <catch2/catch.hpp>

#include <ML/Activation.h>
#include <ML/Utils.h>

TEST_CASE("Linear Activation Function") {
	const ml::ActivationFunc_Linear func(3.0);

	REQUIRE(func.GetTypeName() == ml::EActivationFunction::Linear);

	const ml::Vector a = { 1.0, 2.0, 3.0 };
	const ml::Vector fa = func.F(a);
	const ml::Vector dfa = func.FD(a);

	REQUIRE((fa == ml::Vector({ 3.0, 6.0, 9.0 })));
	REQUIRE((dfa == ml::Vector({ 3.0, 3.0, 3.0 })));
}

TEST_CASE("Sigmoid Activation Function") {
	const ml::ActivationFunc_Sigmoid func(1.0);

	REQUIRE(func.GetTypeName() == ml::EActivationFunction::Sigmoid);

	const ml::Vector a = { 0.5, 1.0, -1.0 };
	const ml::Vector fa = func.F(a);
	const ml::Vector dfa = func.FD(a);

	REQUIRE((fa == ml::Vector({ 0.6224593312, 0.7310585786, 0.2689414214 })));
	REQUIRE((dfa == ml::Vector({ 0.2350037122, 0.1966119332, 0.1966119332 })));
}

TEST_CASE("ReLU Activation Function") {
	const ml::ActivationFunc_ReLU func;

	REQUIRE(func.GetTypeName() == ml::EActivationFunction::ReLU);

	const ml::Vector a = { -1.0, 1.0, 2.0 };
	const ml::Vector fa = func.F(a);
	const ml::Vector dfa = func.FD(a);

	REQUIRE((fa == ml::Vector({ 0.0, 1.0, 2.0 })));
	REQUIRE((dfa == ml::Vector({ 0.0, 1.0, 1.0 })));
}

TEST_CASE("LeakyReLU Activation Function") {
	const ml::ActivationFunc_LeakyReLU func(0.01);

	REQUIRE(func.GetTypeName() == ml::EActivationFunction::LeakyReLU);

	const ml::Vector a = { -1.0, 1.0, 2.0 };
	const ml::Vector fa = func.F(a);
	const ml::Vector dfa = func.FD(a);

	REQUIRE((fa == ml::Vector({ -0.01, 1.0, 2.0 })));
	REQUIRE((dfa == ml::Vector({ 0.01, 1.0, 1.0 })));
}

TEST_CASE("Softsign Activation Function") {
	const ml::ActivationFunc_Softsign func;

	REQUIRE(func.GetTypeName() == ml::EActivationFunction::Softsign);

	const ml::Vector a = { -1.0, 1.0, 3.0 };
	const ml::Vector fa = func.F(a);
	const ml::Vector dfa = func.FD(a);

	REQUIRE((fa == ml::Vector({ -0.5, 0.5, 0.75 })));
	REQUIRE((dfa == ml::Vector({ 0.25, 0.25, 0.0625 })));
}

TEST_CASE("Softplus Activation Function") {
	const ml::ActivationFunc_Softplus func;

	REQUIRE(func.GetTypeName() == ml::EActivationFunction::Softplus);

	const ml::Vector a = { -1.0, 0.0, 1.0 };
	const ml::Vector fa = func.F(a);
	const ml::Vector dfa = func.FD(a);

	REQUIRE((fa == ml::Vector({ 0.3132616875, 0.6931471806, 1.313261688 })));
	REQUIRE((dfa == ml::Vector({ 0.2689414214, 0.5, 0.7310585786 })));
}

TEST_CASE("TanH Activation Function") {
	const ml::ActivationFunc_TanH func;

	REQUIRE(func.GetTypeName() == ml::EActivationFunction::TanH);

	const ml::Vector a = { -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0 };
	const ml::Vector fa = func.F(a);
	const ml::Vector dfa = func.FD(a);

	REQUIRE((fa == ml::Vector({ -0.9950547537, -0.9640275801, -0.761594156, 0.0, 0.761594156, 0.9640275801, 0.9950547537 })));
	REQUIRE((dfa == ml::Vector({ 0.009866037166, 0.07065082485, 0.4199743416, 1.0, 0.4199743416, 0.07065082485, 0.009866037166 })));
}

TEST_CASE("Softmax Activation Function") {
	const ml::ActivationFunc_Softmax func;

	REQUIRE(func.GetTypeName() == ml::EActivationFunction::Softmax);

	const ml::Vector a = { -1.0, 1.0, 2.0 };
	const ml::Vector fa = func.F(a);

	REQUIRE((fa == ml::Vector({ 0.03511902695934, 0.25949646034242, 0.70538451269824 })));
}