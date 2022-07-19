#include <ML/Network.h>
#include <ML/Utils.h>

#include <assert.h>

ml::Network::Network(ELossFunction lossFunction) 
	: m_LossFunction(lossFunction) {
}

void ml::Network::AddLayer(uint64_t neuronCount, const ActivationFunc* activationFunction /*= nullptr*/) {
	if (m_Layers.empty()) {
		m_Layers.emplace_back(Layer(neuronCount));
	} else {
		m_Layers.emplace_back(Layer(m_Layers.back(), neuronCount, activationFunction));
	}
}

void ml::Network::FillSNorm() {
	for (size_t l = 1; l < m_Layers.size(); ++l) {
		Layer& layer = m_Layers[l];
		Matrix& weights = layer.GetWeights();
		Vector& biases = layer.GetBiases();

		for (uint64_t i = 0; i < weights.Rows(); ++i) {
			for (uint64_t j = 0; j < weights.Cols(); ++j) {
				const double z = RndRange(-1.0, 1.0);
				weights[i][j] = exp(-3.1415926 * z * z);
			}
		}

		for (uint64_t i = 0; i < biases.Length(); ++i) {
			const double z = RndRange(-1.0, 1.0);
			biases[i] = exp(-3.1415926 * z * z);
		}
	}
}

void ml::Network::FillWeights(double value) {
	for (size_t l = 1; l < m_Layers.size(); ++l) {
		Layer& layer = m_Layers[l];
		Matrix& weights = layer.GetWeights();
		Vector& biases = layer.GetBiases();

		for (uint64_t i = 0; i < weights.Rows(); ++i) {
			for (uint64_t j = 0; j < weights.Cols(); ++j) {
				weights[i][j] = value;
			}
		}

		for (uint64_t i = 0; i < biases.Length(); ++i) {
			biases[i] = value;
		}
	}
}

void ml::Network::SetInput(const Vector& input) {
	m_Layers[0].GetNeurons() = input;
}

void ml::Network::SetInput(const double* values, uint64_t count) {
	Vector input(count);
	for (uint64_t i = 0; i < count; ++i) {
		input[i] = values[i];
	}
	SetInput(input);
}

auto ml::Network::ForwardPropagate() -> const Vector& {
	for (size_t i = 1; i < m_Layers.size(); ++i) {
		m_Layers[i].ForwardPropagate(m_Layers[i - 1]);
	}

	return m_Layers.back().GetNeurons();
}

auto ml::Network::GetLayers() const ->const std::vector<Layer>& {
	return m_Layers;
}

auto ml::Network::GetLayers()->std::vector<Layer>& {
	return m_Layers;
}

auto ml::Network::GetLossFunction() const->ELossFunction {
	return m_LossFunction;
}

auto ml::Learning::CalculateGradient(const Network& network, const Vector& expected) const->std::vector<LayerGradient> {
	const std::vector<Layer>& layers = network.GetLayers();
	std::vector<LayerGradient> layerGradients(layers.size());

	const uint64_t layerCount = layers.size();
	const uint64_t lastIdx = layerCount - 1;
	const uint64_t firstIdx = 0;

	Vector dEda;

	for (uint64_t lIdx = lastIdx; lIdx > firstIdx; --lIdx) {
		const Layer& layer = layers[lIdx];
		const Layer& layerPrev = layers[lIdx - 1];

		Matrix& wGrad = layerGradients[lIdx].WGrad;
		Vector& bGrad = layerGradients[lIdx].BGrad;
		Vector dEdz;

		const ActivationFunc* activation = layer.GetActivationFunc();

		if (lIdx == lastIdx) {
			const LossRegressionFunc* lossRegrFunc = m_LossRegressionProvider.GetFunc(activation->GetTypeName(), network.GetLossFunction());
			assert(lossRegrFunc != nullptr);

			// Calculate dEdz
			dEdz = lossRegrFunc->Calculate_dEdz(layerPrev, layer, expected);
		} else {

			// Calculate z
			const Vector z = layer.GetWeights() * layerPrev.GetNeurons() + layer.GetBiases();

			// Calculate z derivative
			const Vector zDer = activation->FD(z);

			// Calculate dEdz
			dEdz = dEda * zDer;
		}

		// Calculate dEdw for current layer
		wGrad = dEdz & layerPrev.GetNeurons();

		// Calculate dEdb for current layer
		bGrad = dEdz;

		if (lIdx > 1) {
			// Calculate dEda for previous layer
			dEda = dEdz * layer.GetWeights();
		}
	}

	return layerGradients;
}

auto ml::Learning::CalculateGradient(const Network& network, const double* expValues, uint64_t expCount) const->std::vector<LayerGradient> {
	Vector expected(expCount);
	for (uint64_t i = 0; i < expCount; ++i) {
		expected[i] = expValues[i];
	}
	return CalculateGradient(network, expected);
}

void ml::Learning::Learn(Network& network, const std::vector<LayerGradient>& grad, double alpha) {
	auto& layers = network.GetLayers();

	const uint64_t layerCount = grad.size();

	for (uint64_t i = 1; i < layerCount; ++i) {
		layers[i].GetWeights() -= grad[i].WGrad * alpha;
		layers[i].GetBiases() -= grad[i].BGrad * alpha;
	}
}