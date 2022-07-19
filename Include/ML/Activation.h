#pragma once

#include <ML/Vector.h>
#include <ML/Enums.h>

namespace ml {
	class Layer;
}

namespace ml {
	class ActivationFunc {
	public:
		virtual ~ActivationFunc() {}

		virtual bool CanBeUsedInHiddenLayer() const = 0;
		virtual auto GetTypeName() const->ml::EActivationFunction = 0;

		virtual auto F(const Vector& z) const->Vector = 0;
		virtual auto FD(const Vector& z) const->Vector = 0;
	};
}

namespace ml {
	class ActivationFunc_Linear : public ActivationFunc {
	public:
		ActivationFunc_Linear(double alpha = 1.0);

		bool CanBeUsedInHiddenLayer() const override { return true; }
		auto GetTypeName() const->ml::EActivationFunction override { return ml::EActivationFunction::Linear; }

		auto GetAlpha() const -> double { return m_Alpha; }

		auto F(const Vector& z) const->Vector override;
		auto FD(const Vector& z) const->Vector override;

	private:
		double m_Alpha;
	};

	class ActivationFunc_Sigmoid : public ActivationFunc {
	public:
		ActivationFunc_Sigmoid(double alpha = 1.0);

		bool CanBeUsedInHiddenLayer() const override { return true; }
		auto GetTypeName() const->ml::EActivationFunction override { return ml::EActivationFunction::Sigmoid; }

		auto F(const Vector& z) const->Vector override;
		auto FD(const Vector& z) const->Vector override;

	private:
		double m_Alpha;
	};

	class ActivationFunc_ReLU : public ActivationFunc {
	public:
		bool CanBeUsedInHiddenLayer() const override { return true; }
		auto GetTypeName() const->ml::EActivationFunction override { return ml::EActivationFunction::ReLU; }

		auto F(const Vector& z) const->Vector override;
		auto FD(const Vector& z) const->Vector override;
	};

	class ActivationFunc_LeakyReLU : public ActivationFunc {
	public:
		ActivationFunc_LeakyReLU(double alpha = 1.0);

		bool CanBeUsedInHiddenLayer() const override { return true; }
		auto GetTypeName() const->ml::EActivationFunction override { return ml::EActivationFunction::LeakyReLU; }

		auto F(const Vector& z) const->Vector override;
		auto FD(const Vector& z) const->Vector override;

	private:
		double m_Alpha;
	};

	class ActivationFunc_Softsign : public ActivationFunc {
	public:
		bool CanBeUsedInHiddenLayer() const override { return true; }
		auto GetTypeName() const->ml::EActivationFunction override { return ml::EActivationFunction::Softsign; }

		auto F(const Vector& z) const->Vector override;
		auto FD(const Vector& z) const->Vector override;
	};

	class ActivationFunc_Softplus : public ActivationFunc {
	public:
		bool CanBeUsedInHiddenLayer() const override { return true; }
		auto GetTypeName() const->ml::EActivationFunction override { return ml::EActivationFunction::Softplus; }

		auto F(const Vector& z) const->Vector override;
		auto FD(const Vector& z) const->Vector override;
	};

	class ActivationFunc_TanH : public ActivationFunc {
	public:
		bool CanBeUsedInHiddenLayer() const override { return true; }
		auto GetTypeName() const->ml::EActivationFunction override { return ml::EActivationFunction::TanH; }

		auto F(const Vector& z) const->Vector override;
		auto FD(const Vector& z) const->Vector override;
	};
}

namespace ml {
	class ActivationFunc_Softmax : public ActivationFunc {
	public:
		bool CanBeUsedInHiddenLayer() const override { return false; }
		auto GetTypeName() const->ml::EActivationFunction override { return ml::EActivationFunction::Softmax; }

		auto F(const Vector& z) const->Vector override;
		auto FD(const Vector& z) const->Vector override;
	};
}