#include <catch2/catch.hpp>

#include <ML/Layer.h>
#include <ML/Utils.h>
#include <ML/Activation.h>
#include <ML/LossRegression.h>

TEST_CASE("Softmax and CategoricalCrossentropy") {
	ml::LossRegressionProvider provider;
	const ml::LossRegressionFunc* func = provider.GetFunc(
		ml::EActivationFunction::Softmax,
		ml::ELossFunction::CategoricalCrossentropy
	);

	ml::Layer a(2);
	ml::Layer b(a, 2, new ml::ActivationFunc_Softmax);

	a.GetNeurons() = { 1.0, 2.0 };

	b.GetWeights() = { {0.5, 0.2}, {0.2, 0.5} };
	b.GetBiases() = { 0.0, 0.0 };

	b.ForwardPropagate(a);
	REQUIRE((b.GetNeurons() == ml::Vector({ 0.42555748318834102, 0.57444251681165892 })));

	const ml::Vector dEdz = func->Calculate_dEdz(a, b, ml::Vector({ 1.0, 0.0 }));

	const ml::Matrix wGrad = dEdz & a.GetNeurons();
	const ml::Vector bGrad = dEdz;

	REQUIRE((dEdz == ml::Vector({
		0.42555748318834102 - 1.0,
		0.57444251681165892 - 0.0
		})));

	REQUIRE((wGrad == ml::Matrix({
		{ (0.42555748318834102 - 1.0) * 1.0, (0.42555748318834102 - 1.0) * 2.0 },
		{ (0.57444251681165892 - 0.0) * 1.0, (0.57444251681165892 - 0.0) * 2.0 },
	})));

	REQUIRE((bGrad == ml::Vector({ 
		0.42555748318834102 - 1.0,
		0.57444251681165892 - 0.0 
	})));
}