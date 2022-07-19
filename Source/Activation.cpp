#include <ML/Activation.h>

#include <assert.h>
#include <math.h>

ml::ActivationFunc_Linear::ActivationFunc_Linear(double alpha) 
	: m_Alpha(alpha) {}

auto ml::ActivationFunc_Linear::F(const Vector& z) const->Vector {
	ml::Vector v(z.Length());

	for (uint64_t i = 0; i < v.Length(); ++i) {
		v[i] = z[i] * m_Alpha;
	}

	return v;
}

auto ml::ActivationFunc_Linear::FD(const Vector& z) const->Vector {
	ml::Vector v(z.Length());

	for (uint64_t i = 0; i < v.Length(); ++i) {
		v[i] = m_Alpha;
	}

	return v;
}



ml::ActivationFunc_Sigmoid::ActivationFunc_Sigmoid(double alpha /*= 1.0*/) 
	: m_Alpha(alpha) {
}

auto ml::ActivationFunc_Sigmoid::F(const Vector& z) const->Vector {
	ml::Vector v(z.Length());

	for (uint64_t i = 0; i < v.Length(); ++i) {
		v[i] = 1.0 / (1.0 + exp(-m_Alpha * z[i]));
	}

	return v;
}

auto ml::ActivationFunc_Sigmoid::FD(const Vector& z) const->Vector {
	ml::Vector v(z.Length());

	for (uint64_t i = 0; i < v.Length(); ++i) {
		const double f = 1.0 / (1.0 + exp(-m_Alpha * z[i]));

		v[i] = m_Alpha * f * (1.0 - f);
	}

	return v;
}





auto ml::ActivationFunc_ReLU::F(const Vector& z) const->Vector {
	ml::Vector v(z.Length());

	for (uint64_t i = 0; i < v.Length(); ++i) {
		v[i] = (z[i] > 0.0) ? z[i] : 0.0;
	}

	return v;
}

auto ml::ActivationFunc_ReLU::FD(const Vector& z) const->Vector {
	ml::Vector v(z.Length());

	for (uint64_t i = 0; i < v.Length(); ++i) {
		v[i] = (z[i] > 0.0) ? (1.0) : (0.0);
	}


	return v;
}




ml::ActivationFunc_LeakyReLU::ActivationFunc_LeakyReLU(double alpha)
	: m_Alpha(alpha) {}

auto ml::ActivationFunc_LeakyReLU::F(const Vector& z) const->Vector {
	ml::Vector v(z.Length());

	for (uint64_t i = 0; i < v.Length(); ++i) {
		v[i] = (z[i] > 0.0) ? z[i] : (m_Alpha * z[i]);
	}

	return v;
}

auto ml::ActivationFunc_LeakyReLU::FD(const Vector& z) const->Vector {
	ml::Vector v(z.Length());

	for (uint64_t i = 0; i < v.Length(); ++i) {
		v[i] = (z[i] > 0.0) ? 1.0 : m_Alpha;
	}

	return v;
}




auto ml::ActivationFunc_Softsign::F(const Vector& z) const->Vector {
	ml::Vector v(z.Length());

	for (uint64_t i = 0; i < v.Length(); ++i) {
		v[i] = z[i] / (1.0 + fabs(z[i]));
	}

	return v;
}

auto ml::ActivationFunc_Softsign::FD(const Vector& z) const->Vector {
	ml::Vector v(z.Length());

	for (uint64_t i = 0; i < v.Length(); ++i) {
		v[i] = 1.0 / pow((1 + fabs(z[i])), 2.0);
	}

	return v;
}





auto ml::ActivationFunc_Softplus::F(const Vector& z) const->Vector {
	ml::Vector v(z.Length());

	for (uint64_t i = 0; i < v.Length(); ++i) {
		v[i] = log(1.0 + exp(z[i]));
	}

	return v;
}

auto ml::ActivationFunc_Softplus::FD(const Vector& z) const->Vector {
	ml::Vector v(z.Length());

	for (uint64_t i = 0; i < v.Length(); ++i) {
		v[i] = 1.0 / (1.0 + exp(-z[i]));
	}

	return v;
}




auto ml::ActivationFunc_TanH::F(const Vector& z) const->Vector {
	ml::Vector v(z.Length());

	for (uint64_t i = 0; i < v.Length(); ++i) {
		v[i] = (exp(z[i]) - exp(-z[i])) / (exp(z[i]) + exp(-z[i]));
	}

	return v;
}

auto ml::ActivationFunc_TanH::FD(const Vector& z) const->Vector {
	ml::Vector v(z.Length());

	for (uint64_t i = 0; i < v.Length(); ++i) {
		const double f = (exp(z[i]) - exp(-z[i])) / (exp(z[i]) + exp(-z[i]));

		v[i] = 1.0 - f * f;
	}

	return v;
}






auto ml::ActivationFunc_Softmax::F(const Vector& z) const->Vector {
	ml::Vector v(z.Length());

	double sum = 0.0;
	for (uint64_t i = 0; i < v.Length(); ++i) {
		sum += exp(z[i]);
	}

	for (uint64_t i = 0; i < v.Length(); ++i) {
		v[i] = exp(z[i]) / sum;
	}

	return v;
}

auto ml::ActivationFunc_Softmax::FD(const Vector& z) const->Vector {
	assert(false); // Can't be used for hidden layer
	return z;
}