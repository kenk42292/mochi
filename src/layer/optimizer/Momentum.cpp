/*
 * Momentum.cpp
 *
 *  Created on: Jan 16, 2017
 *      Author: ken
 */

#include "Momentum.hpp"

Momentum::Momentum(double eta, double gamma=0.9): mEta(eta), cacheInitialized(false), mGamma(gamma) {}

Momentum::~Momentum() {}

arma::field<arma::Cube<double>> Momentum::delta(
		const arma::field<arma::Cube<double>>& gradients,
		unsigned int batchSize) {
	if (!cacheInitialized) {
		cacheInitialized = true;
		mCache = gradients;
	}
	arma::field<arma::Cube<double>> paramChange(gradients.size());
	for (unsigned int i = 0; i < gradients.size(); ++i) {
		mCache[i] = mGamma*mCache[i] + (1.0-mGamma)*gradients[i];
		paramChange[i] = (mEta / batchSize) * mCache[i];
	}
	return paramChange;
}

