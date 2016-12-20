/*
 * Adagrad.cpp
 *
 *  Created on: Dec 19, 2016
 *      Author: ken
 */

#include "Adagrad.hpp"

Adagrad::Adagrad(double eta) :
		mEta(eta), cacheInitialized(false), eps(0.00001) {
}

Adagrad::~Adagrad() {
	// TODO Auto-generated destructor stub
}

arma::field<arma::Cube<double>> Adagrad::delta(
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
		mCache[i] += gradients[i] % gradients[i];
		paramChange[i] = (mEta / batchSize) * gradients[i]
				/ (arma::sqrt(mCache[i]) + eps);
	}
	return paramChange;
}


