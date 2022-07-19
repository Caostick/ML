#pragma once

#include <ML/Vector.h>
#include <ML/Layer.h>
#include <ML/Enums.h>

namespace ml {
	class LossRegressionFunc {
	public:
		virtual ~LossRegressionFunc() {}

		virtual auto Calculate_dEdz(const Layer& layerPrev, const Layer& layerLast, const Vector& expected) const -> Vector = 0;
	};

	class LossRegressionProvider {
	public:
		LossRegressionProvider();
		~LossRegressionProvider();

		auto GetFunc(EActivationFunction aFunc, ELossFunction lFunc) const ->const LossRegressionFunc*;
	private:
		LossRegressionFunc* m_Functions[int(EActivationFunction::NUM)][int(ELossFunction::NUM)];
	};
}

namespace ml {
	class LossRegressionFunc_Softmax_CategoricalCrossentropy : public LossRegressionFunc {
	public:
		auto Calculate_dEdz(const Layer& layerPrev, const Layer& layerLast, const Vector& expected) const -> Vector override;
	};

	class LossRegressionFunc_AnySeparate_MeanSquaredError : public LossRegressionFunc {
	public:
		auto Calculate_dEdz(const Layer& layerPrev, const Layer& layerLast, const Vector& expected) const->Vector override;
	};
}