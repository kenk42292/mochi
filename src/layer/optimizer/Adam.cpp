/*
 * Adam.cpp
 *
 *  Created on: Jan 17, 2017
 *      Author: ken
 */

#include "Adam.hpp"

Adam::Adam(double eta, double b1=0.9, double b2=0.999): mEta(eta), cacheInitialized(false), b1(b1), b2(b2), t(0) {}

Adam::~Adam() {}

arma::field<arma::Cube<double>> Adam::delta(
		const arma::field<arma::Cube<double>>& gradients,
		unsigned int batchSize) {
	if (!cacheInitialized) {
		cacheInitialized = true;
		mM = arma::field<arma::Cube<double>>(gradients.size());
		mV = arma::field<arma::Cube<double>>(gradients.size());
		for (unsigned int i=0; i<gradients.size(); ++i) {
			unsigned int nr=gradients[i].n_rows, nc=gradients[i].n_cols, ns=gradients[i].n_slices;
			mM[i] = arma::Cube<double>(nr, nc, ns, arma::fill::zeros);
			mV[i] = arma::Cube<double>(nr, nc, ns, arma::fill::zeros);
		}
	}
	t+=1;
	arma::field<arma::Cube<double>> paramChanges(gradients.size());
	for (unsigned int i = 0; i < gradients.size(); ++i) {
		mM[i] = b1*mM[i] + (1.0-b1)*gradients[i];
		mV[i] = b2*mV[i] + (1.0-b2)*gradients[i]%gradients[i];
		paramChanges[i] = (mEta/batchSize)*(sqrt(1-pow(b2,t))/(1-pow(b1,t)))*(mM[i]/(arma::sqrt(mV[i])+eps));
	}
	return paramChanges;
}

