/*
 * GradientDescent.cpp
 *
 *  Created on: Dec 19, 2016
 *      Author: ken
 */

#include "GradientDescent.hpp"

GradientDescent::GradientDescent(double eta): mEta(eta) {}

GradientDescent::~GradientDescent() {}

arma::field<arma::Cube<double>> GradientDescent::delta(const arma::field<arma::Cube<double>>& gradients, unsigned int batchSize) {
	arma::field<arma::Cube<double>> paramChanges(gradients.size());
	for (unsigned int i=0; i<gradients.size(); ++i) {
		paramChanges[i] = (mEta/batchSize)*gradients[i];
	}
	return paramChanges;
};

