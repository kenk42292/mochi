/*
 * RMSProp.cpp
 *
 *  Created on: Dec 19, 2016
 *      Author: ken
 */

#include "RMSProp.hpp"

RMSProp::RMSProp(double eta): mEta(eta) {}

RMSProp::~RMSProp() {}

arma::field<arma::Cube<double>> RMSProp::delta(const arma::field<arma::Cube<double>>& gradients, unsigned int batchSize) {
	arma::field<arma::Cube<double>> paramChanges(gradients.size());
	for (unsigned int i=0; i<gradients.size(); ++i) {
		paramChanges[i] = (mEta/batchSize)*gradients[i];
	}
	return paramChanges;
};
