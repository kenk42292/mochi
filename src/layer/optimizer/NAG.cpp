/*
 * NAG.cpp
 *
 *  Created on: Jan 17, 2017
 *      Author: ken
 */

#include "NAG.hpp"

NAG::NAG(double eta, double gamma=0.9): mEta(eta), cacheInitialized(false), mGamma(gamma) {}

NAG::~NAG() {}

arma::field<arma::Cube<double>> NAG::delta(
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
	arma::field<arma::Cube<double>> vprev = mMomentum;
	for (unsigned int i = 0; i < gradients.size(); ++i) {
		mMomentum[i] = mGamma*mMomentum[i] + (mEta / batchSize) * gradients[i];
	}
	arma::field<arma::Cube<double>> paramChanges(gradients.size());
	for (unsigned int i=0; i<gradients.size(); ++i) {
		paramChanges[i] = -mGamma*vprev[i] + (1.0+mGamma)*mMomentum[i];
	}
	return paramChanges;
}

