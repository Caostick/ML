#include <ML/LossRegression.h>
#include <ML/Utils.h>

#include <assert.h>

ml::LossRegressionProvider::LossRegressionProvider() {
	for (int i = 0; i< int(EActivationFunction::NUM); ++i) {
		for (int j = 0; j< int(ELossFunction::NUM); ++j) {
			m_Functions[i][j] = nullptr;
		}
	}

	m_Functions[int(EActivationFunction::Softmax)][int(ELossFunction::CategoricalCrossentropy)] = new LossRegressionFunc_Softmax_CategoricalCrossentropy;
	m_Functions[int(EActivationFunction::Linear)][int(ELossFunction::MeanSquaredError)] = new LossRegressionFunc_AnySeparate_MeanSquaredError;
	m_Functions[int(EActivationFunction::Sigmoid)][int(ELossFunction::MeanSquaredError)] = new LossRegressionFunc_AnySeparate_MeanSquaredError;
}

ml::LossRegressionProvider::~LossRegressionProvider() {
	for (int i = 0; i< int(EActivationFunction::NUM); ++i) {
		for (int j = 0; j< int(ELossFunction::NUM); ++j) {
			delete m_Functions[i][j];
		}
	}
}

auto ml::LossRegressionProvider::GetFunc(EActivationFunction aFunc, ELossFunction lFunc)const -> const LossRegressionFunc* {
	LossRegressionFunc* func = m_Functions[int(aFunc)][int(lFunc)];

	assert(func != nullptr);

	return func;
}




auto ml::LossRegressionFunc_Softmax_CategoricalCrossentropy::Calculate_dEdz(
	const Layer& /*layerPrev*/,
	const Layer& layerLast,
	const Vector& expected) const->Vector {

	Vector dEdz(layerLast.GetNeuronCount());

	for (uint64_t i = 0; i < dEdz.Length(); ++i) {
		dEdz[i] = layerLast.GetNeurons()[i] - expected[i];
	}

	return dEdz;
}

auto ml::LossRegressionFunc_AnySeparate_MeanSquaredError::Calculate_dEdz(
	const Layer& layerPrev,
	const Layer& layerLast,
	const Vector& expected) const->Vector {

	const double invN = 1.0 / layerLast.GetNeuronCount();

	Vector dEda(layerLast.GetNeuronCount());
	Vector dEdz(layerLast.GetNeuronCount());

	// Calculate dEda
	for (uint64_t i = 0; i < dEda.Length(); ++i) {
		dEda[i] = 2.0 * invN * (layerLast.GetNeurons()[i] - expected[i]);
	}

	// Calculate z
	const Vector z = layerLast.GetWeights() * layerPrev.GetNeurons() + layerLast.GetBiases();

	// Calculate z derivative
	const Vector zDer = layerLast.GetActivationFunc()->FD(z);

	// Calculate dEdz
	dEdz = dEda * zDer;

	return dEdz;
}