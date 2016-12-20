/*
 * Optimizer.cpp
 *
 *  Created on: Dec 19, 2016
 *      Author: ken
 */

#include "Optimizer.hpp"

Optimizer::Optimizer() {}

Optimizer::~Optimizer() {}

arma::field<arma::Cube<double>> Optimizer::delta(const arma::field<arma::Cube<double>>& gradients, unsigned int batchSize) {
	return gradients;
};
