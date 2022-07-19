#pragma once

#include <ML/Vector.h>
#include <ML/Matrix.h>
#include <ML/Activation.h>

namespace ml {
	class Layer {
	public:
		Layer(uint64_t neuronCount, const ActivationFunc* activationFunc = nullptr);
		Layer(const Layer& layerIn, uint64_t neuronCount, const ActivationFunc* activationFunc = nullptr);
		Layer(Layer&& other);
		~Layer();

		auto GetNeuronCount() const->uint64_t;

		auto GetNeurons() -> Vector&;
		auto GetNeurons() const -> const Vector&;

		auto GetBiases()->Vector&;
		auto GetBiases() const -> const Vector&;

		auto GetWeights()->Matrix&;
		auto GetWeights() const -> const Matrix&;

		auto GetActivationFunc() const -> const ActivationFunc*;

		void ForwardPropagate(const Layer& layerPrev);

	private:
		Vector m_Neurons;
		Vector m_Biases;
		Matrix m_Weights;

		const ActivationFunc* m_ActivationFunc;
	};

	struct LayerGradient {
		LayerGradient() = default;
		LayerGradient(const LayerGradient&) = default;
		LayerGradient(LayerGradient&&) = default;

		LayerGradient& operator = (const LayerGradient&) = default;
		LayerGradient& operator = (LayerGradient&&) = default;

		Matrix WGrad;
		Vector BGrad;
	};
}