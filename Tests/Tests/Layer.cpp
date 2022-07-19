#include <catch2/catch.hpp>

#include <ML/Layer.h>
#include <ML/Utils.h>

#include <vector>

TEST_CASE("Layer Construction") {
	ml::Layer a(2);
	ml::Layer b(a, 3);

	REQUIRE(a.GetNeuronCount() == 2);
	REQUIRE(a.GetNeurons().Length() == 2);
	REQUIRE(a.GetBiases().Length() == 2);
	REQUIRE(a.GetWeights().Cols() == 0);
	REQUIRE(a.GetWeights().Rows() == 0);

	REQUIRE(b.GetNeuronCount() == 3);
	REQUIRE(b.GetNeurons().Length() == 3);
	REQUIRE(b.GetBiases().Length() == 3);
	REQUIRE(b.GetWeights().Cols() == 2);
	REQUIRE(b.GetWeights().Rows() == 3);

	std::vector<ml::Layer> net;
	net.emplace_back(ml::Layer(2));
	net.emplace_back(ml::Layer(net.back(), 3));

	REQUIRE(net[0].GetNeuronCount() == 2);
	REQUIRE(net[0].GetNeurons().Length() == 2);
	REQUIRE(net[0].GetBiases().Length() == 2);
	REQUIRE(net[0].GetWeights().Cols() == 0);
	REQUIRE(net[0].GetWeights().Rows() == 0);

	REQUIRE(net[1].GetNeuronCount() == 3);
	REQUIRE(net[1].GetNeurons().Length() == 3);
	REQUIRE(net[1].GetBiases().Length() == 3);
	REQUIRE(net[1].GetWeights().Cols() == 2);
	REQUIRE(net[1].GetWeights().Rows() == 3);
}

TEST_CASE("Simple Forward Propagation") {
	ml::Layer a(1);
	ml::Layer b(a, 1);

	a.GetNeurons() = { 2.0 };

	b.GetWeights() = { {3.0} };
	b.GetBiases() = { 4.0 };

	b.ForwardPropagate(a);

	REQUIRE((b.GetNeurons() == ml::Vector({ 10.0 })));
}

TEST_CASE("Complex Forward Propagation") {
	ml::Layer a(2);
	ml::Layer b(a, 3);
	ml::Layer c(b, 2);

	a.GetNeurons() = { 1.0, 2.0 };

	b.GetWeights() = { {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0} };
	b.GetBiases() = { 3.0, 2.0, 1.0 };

	c.GetWeights() = { {6.0, 4.0, 2.0}, {5.0, 3.0, 1.0} };
	c.GetBiases() = {1.0, 2.0};

	b.ForwardPropagate(a);
	REQUIRE((b.GetNeurons() == ml::Vector({ 8.0, 13.0, 18.0 })));

	c.ForwardPropagate(b);
	REQUIRE((c.GetNeurons() == ml::Vector({ 137.0, 99.0 })));
}

TEST_CASE("Forward Propagation With Activation") {
	ml::Layer a(2);
	ml::Layer b(a, 3, new ml::ActivationFunc_Softmax);

	a.GetNeurons() = { 1.0, 2.0 };

	b.GetWeights() = { {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0} };
	b.GetBiases() = { 3.0, 2.0, 1.0 };

	b.ForwardPropagate(a);
	REQUIRE((b.GetNeurons() == ml::Vector({ 4.5094041236355E-5, 0.0066925491165893, 0.99326235684217 })));
}