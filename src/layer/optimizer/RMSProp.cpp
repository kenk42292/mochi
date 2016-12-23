/*
 * RMSProp.cpp
 *
 *  Created on: Dec 19, 2016
 *      Author: ken
 */

#include "RMSProp.hpp"

RMSProp::RMSProp(double eta, double gamma): mEta(eta), mGamma(gamma), cacheInitialized(false), eps(0.00001) {}

RMSProp::~RMSProp() {}

arma::field<arma::Cube<double>> RMSProp::delta(
		const arma::field<arma::Cube<double>>& gradients,
		unsigned int batchSize) {
	if (!cacheInitialized) {
		cacheInitialized = true;
		mCache = arma::field<arma::Cube<double>>(gradients.size());
		for (unsigned int i = 0; i < gradients.size(); ++i) {
			mCache[i] = arma::Cube<double>(gradients[i].n_rows,
					gradients[i].n_cols, gradients[i].n_slices,
					arma::fill::zeros);
		}
	}
	arma::field<arma::Cube<double>> paramChange(gradients.size());
	for (unsigned int i = 0; i < gradients.size(); ++i) {
		mCache[i] = mGamma*mCache[i]+(1.0-mGamma)*gradients[i] % gradients[i];
		paramChange[i] = (mEta / batchSize) * gradients[i]
				/ (arma::sqrt(mCache[i]) + eps);
	}
	return paramChange;
}
