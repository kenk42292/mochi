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
		mMomentum = arma::field<arma::Cube<double>>(gradients.size());
		for (unsigned int i=0; i<gradients.size(); ++i) {
			unsigned int nr=gradients[i].n_rows, nc=gradients[i].n_cols, ns=gradients[i].n_slices;
			mMomentum[i] = arma::Cube<double>(nr, nc, ns, arma::fill::zeros);
		}
	}
	for (unsigned int i = 0; i < gradients.size(); ++i) {
		mMomentum[i] = mGamma*mMomentum[i] + (mEta / batchSize) * gradients[i];
	}
	return mMomentum;
}

