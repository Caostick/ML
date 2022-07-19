#pragma once

namespace ml {
	enum class EActivationFunction {
		None,
		Linear,
		Sigmoid,
		ReLU,
		LeakyReLU,
		Softsign,
		Softplus,
		TanH,
		Softmax,

		NUM
	};

	enum class ELossFunction {
		None,
		CategoricalCrossentropy,
		BinaryCrossentropy,
		SquaredHinge,
		MeanSquaredError,
		MeanAbsoluteError,
		MeanSquaredLogarithmicError,
		Poisson,
		NUM
	};
}









