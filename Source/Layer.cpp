#include <ML/Layer.h>
#include <ML/Utils.h>

#include <xutility>

ml::Layer::Layer(uint64_t neuronCount, const ActivationFunc* activationFunc /*= nullptr*/)
	: m_Neurons(neuronCount)
	, m_Biases(neuronCount)
	, m_Weights() 
	, m_ActivationFunc(activationFunc) {
}

ml::Layer::Layer(const Layer& layerIn, uint64_t neuronCount, const ActivationFunc* activationFunc /*= nullptr*/)
	: m_Neurons(neuronCount)
	, m_Biases(neuronCount)
	, m_Weights(layerIn.GetNeuronCount(), neuronCount) 
	, m_ActivationFunc(activationFunc) {
}

ml::Layer::Layer(Layer&& other) 
	: m_Neurons(std::move(other.m_Neurons))
	, m_Biases(std::move(other.m_Biases))
	, m_Weights(std::move(other.m_Weights)) {

	m_ActivationFunc = other.m_ActivationFunc;
	other.m_ActivationFunc = nullptr;
}

ml::Layer::~Layer() {
	delete m_ActivationFunc;
}

auto ml::Layer::GetNeuronCount() const->uint64_t {
	return m_Neurons.Length();
}

auto ml::Layer::GetNeurons()->Vector& {
	return m_Neurons;
}

auto ml::Layer::GetNeurons() const -> const Vector& {
	return m_Neurons;
}

auto ml::Layer::GetBiases()->Vector& {
	return m_Biases;
}

auto ml::Layer::GetBiases() const -> const Vector& {
	return m_Biases;
}

auto ml::Layer::GetWeights()->Matrix& {
	return m_Weights;
}

auto ml::Layer::GetWeights() const -> const Matrix& {
	return m_Weights;
}

auto ml::Layer::GetActivationFunc() const -> const ActivationFunc* {
	return m_ActivationFunc;
}

void ml::Layer::ForwardPropagate(const Layer& layerPrev) {
	m_Neurons = m_Weights * layerPrev.GetNeurons() + m_Biases;

	if (m_ActivationFunc) {
		m_Neurons = m_ActivationFunc->F(m_Neurons);
	}
}