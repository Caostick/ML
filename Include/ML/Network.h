#pragma once

#include <ML/Layer.h>
#include <ML/Enums.h>
#include <ML/LossRegression.h>

#include <stdint.h>
#include <vector>

// @TODO: Optimizations(Adam etc.)
// @TODO: Metrics(Accuracy etc.)

namespace ml {
	class Network {
	public:
		Network(ELossFunction lossFunction);

		void AddLayer(uint64_t neuronCount, const ActivationFunc* activationFunction = nullptr);

		void FillSNorm();
		void FillWeights(double value);

		void SetInput(const Vector& input);
		void SetInput(const double* values, uint64_t count);
		auto ForwardPropagate() -> const Vector&;

		auto GetLayers() const ->const std::vector<Layer>&;
		auto GetLayers() -> std::vector<Layer>&;

		auto GetLossFunction() const->ELossFunction;

	private:
		std::vector<Layer> m_Layers;

		ELossFunction m_LossFunction;
	};

	class Learning {
	public:

		auto CalculateGradient(const Network& network, const Vector& expected) const->std::vector<LayerGradient>;
		auto CalculateGradient(const Network& network, const double* expValues, uint64_t expCount) const->std::vector<LayerGradient>;

		void Learn(Network& network, const std::vector<LayerGradient>& grad, double alpha);

	private:
		LossRegressionProvider m_LossRegressionProvider;
	};
}